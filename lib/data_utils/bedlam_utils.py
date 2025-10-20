import os
import os.path as osp
import argparse
from typing import Dict, List, Tuple, Optional

import cv2
import gc
import numpy as np
import joblib
import torch
from tqdm import tqdm

import sys
sys.path.append('.')

from lib.models import spin
from lib.core.config import TCMR_DB_DIR
from lib.utils.smooth_bbox import get_smooth_bbox_params
from lib.data_utils._feature_extractor import extract_features
from lib.data_utils._kp_utils import smpl_2d_to_common14, convert_kps
from lib.models.smpl import SMPL, SMPL_MODEL_DIR


VIS_THRESH = 0.3
MIN_KP = 6


def _ensure_list(x):
    if isinstance(x, list):
        return x
    if isinstance(x, np.ndarray):
        return x.tolist()
    return [x]


def _to_str(x):
    if isinstance(x, np.ndarray):
        if x.dtype == np.object_ and x.size == 1:
            return str(x.item())
        return str(x)
    return str(x)


def resolve_image_path(image_root: str, npz_path: str, imgname: str) -> str:
    """Resolve BEDLAM image path using common layouts.

    Expected layout used by df_full_body_smpl.py and bedlam_utils.py:
      <image_root>/<scene_name>/png/<imgname>
    where <scene_name> ~= basename(npz_path).replace('.npz','').
    and <imgname> looks like '<seq_name>/seq_xxxxx.png'
    """
    scene_name = osp.splitext(osp.basename(npz_path))[0]
    # normalize separators
    img_rel = imgname.replace('\\', '/')

    cand1 = osp.join(image_root, scene_name, 'png', img_rel)
    if osp.exists(cand1):
        return cand1

    # Fallback: try directly under image_root
    cand2 = osp.join(image_root, img_rel)
    if osp.exists(cand2):
        return cand2

    # As last resort, return path 1 (let caller handle errors)
    return cand1


def group_indices_by_video(imgnames: List[str], subs: Optional[List[str]] = None) -> Dict[str, List[int]]:
    """Group frame indices by a video key inferred from imgnames and subject id.

    Key format: '<seq_name>_<sub>' if subs provided else '<seq_name>'.
    seq_name is the first path component in imgname (e.g., 'seq_000000').
    """
    groups: Dict[str, List[int]] = {}
    for i, nm in enumerate(imgnames):
        p = nm.replace('\\', '/')
        parts = p.split('/')
        seq = parts[0] if len(parts) > 1 else osp.splitext(osp.basename(p))[0]
        sub = _to_str(subs[i]) if (subs is not None and len(subs) > i) else ''
        key = f"{seq}_{sub}" if sub != '' else seq
        groups.setdefault(key, []).append(i)
    return groups


def _to_2d_numeric(arr: np.ndarray) -> np.ndarray:
    """Coerce input to a 2D float32 array [N, D], padding/truncating rows to the same D if needed."""
    arr_np = np.asarray(arr, dtype=object)
    if arr_np.ndim == 2 and arr_np.dtype != object:
        return arr_np.astype(np.float32)
    if arr_np.ndim == 1 and arr_np.dtype == object:
        maxd = 0
        items = []
        for x in arr_np:
            xi = np.asarray(x).ravel()
            items.append(xi)
            maxd = max(maxd, xi.shape[0])
        out = np.zeros((len(items), maxd), dtype=np.float32)
        for i, xi in enumerate(items):
            d = min(maxd, xi.shape[0])
            out[i, :d] = xi[:d]
        return out
    if arr_np.ndim == 1:
        return arr_np.astype(np.float32)[None, :]
    # Fallback
    return np.asarray(arr, dtype=np.float32)


def compute_smpl_joints49(pose: np.ndarray, betas: np.ndarray, transl: np.ndarray, smpl_model: Optional[SMPL] = None) -> np.ndarray:
    """Compute 49-joint SPIN-style SMPL joints from pose/betas/transl.

    - pose: (N,72) axis-angle (global+body)
    - betas: (N,10)
    - transl: (N,3)
    Returns (N,49,3)
    """
    smpl = smpl_model if smpl_model is not None else SMPL(SMPL_MODEL_DIR, batch_size=1, create_transl=False)
    with torch.no_grad():
        p = torch.from_numpy(_to_2d_numeric(pose)).float()
        bt = _to_2d_numeric(betas)
        # Match betas dimension to model shapedirs
        try:
            k = int(smpl.shapedirs.shape[-1])
        except Exception:
            # Fallback to 10
            k = 10
        if bt.shape[1] < k:
            pad = np.zeros((bt.shape[0], k - bt.shape[1]), dtype=bt.dtype)
            bt = np.concatenate([bt, pad], axis=1)
        elif bt.shape[1] > k:
            bt = bt[:, :k]
        b = torch.from_numpy(bt).float()
        t = torch.from_numpy(_to_2d_numeric(transl)).float()
        out = smpl(betas=b, body_pose=p[:, 3:], global_orient=p[:, :3], transl=t)
        j3d = out.joints.cpu().numpy()  # (N,49,3)
    return j3d


def read_data(npz_dir: str, images_dir: str, set_name: str,
              split_file: Optional[str] = None, debug: bool = False, hmr_batch: int = 200) -> Dict[str, np.ndarray]:
    """Build BEDLAM DB from processed .npz files.

    - npz_dir: folder containing BEDLAM .npz exported by df_full_body_smpl.py
    - images_dir: base images folder (contains <scene>/png/<seq>/image_xxx.png)
    - set_name: one of {'train','val','test'}; affects 3D format
    - split_file: optional path to a text file listing scene basenames to include (one per line)
    """
    dataset = {
        'vid_name': [],
        'frame_id': [],
        'joints3D': [],
        'joints2D': [],
        'shape': [],
        'pose': [],
        'bbox': [],
        'img_name': [],
        'features': [],
        # 'valid': [],  # optional
    }

    model = spin.get_pretrained_hmr()
    smpl_model = SMPL(SMPL_MODEL_DIR, batch_size=1, create_transl=False)

    # Collect npz files
    all_npz = [osp.join(npz_dir, f) for f in os.listdir(npz_dir) if f.endswith('.npz')]
    if split_file is not None and osp.isfile(split_file):
        with open(split_file, 'r') as f:
            allowed = {line.strip() for line in f if line.strip()}
        all_npz = [p for p in all_npz if osp.splitext(osp.basename(p))[0] in allowed]

    for npz_path in tqdm(sorted(all_npz), desc=f'BEDLAM {set_name}'):  # per scene
        data = np.load(npz_path, allow_pickle=True)

        imgnames = [
            _to_str(x) for x in _ensure_list(data['imgname'])
        ]
        gtkps = np.asarray(data['gtkps'])  # (N, J(=24), 3)
        pose_cam = np.asarray(data['pose_cam'])  # (N,72)
        betas = np.asarray(data['shape'])  # (N, >=10)
        trans_cam = np.asarray(data['trans_cam'])  # (N,3)
        subs = _ensure_list(data['sub']) if 'sub' in data.files else [''] * len(imgnames)

        # Group indices per video (seq + subject)
        groups = group_indices_by_video(imgnames, subs=subs)

        # Per-scene progress bar (by frames)
        total_frames = sum(len(v) for v in groups.values())
        scene_name = osp.splitext(osp.basename(npz_path))[0]
        scene_pbar = tqdm(total=total_frames, desc=f'scene {scene_name}', leave=False)

        for key, idxs in groups.items():
            # keep original temporal order based on filename
            idxs = sorted(idxs, key=lambda i: imgnames[i])

            # Resolve image paths
            img_paths = [resolve_image_path(images_dir, npz_path, imgnames[i]) for i in idxs]

            # 2D keypoints: source SMPL (24), map to COMMON 14
            j2d_smpl = gtkps[idxs].astype(np.float32)  # (T,24,3)
            # ensure visibility channel exists and reasonable
            if j2d_smpl.shape[-1] == 2:
                vis = np.ones((j2d_smpl.shape[0], j2d_smpl.shape[1], 1), dtype=np.float32)
                j2d_smpl = np.concatenate([j2d_smpl, vis], axis=-1)

            # Use a loose visibility thresh since BEDLAM gt has 1.0 in the last channel
            bbox_params, time_pt1, time_pt2 = get_smooth_bbox_params(j2d_smpl, vis_thresh=0.1, sigma=8)
            c_x = bbox_params[:, 0]
            c_y = bbox_params[:, 1]
            scale = bbox_params[:, 2]
            w = h = 150.0 / scale
            w = h = h * 1.1
            bbox = np.vstack([c_x, c_y, w, h]).T

            # Convert 2D SMPL -> COMMON14 (append neck/headtop placeholders invisible)
            j2d_common = np.stack([smpl_2d_to_common14(kp) for kp in j2d_smpl], axis=0)  # (T,14,3)

            # Slice temporal stable range
            sl = slice(time_pt1, time_pt2)
            img_paths_array = np.array(img_paths)[sl]
            j2d_common = j2d_common[sl]
            bbox = bbox[sl]

            # Pose/shape; ensure dims
            pose = pose_cam[idxs].astype(np.float32)[sl]  # (T,72)
            shape = betas[idxs].astype(np.float32)[sl]
            if shape.ndim == 1:
                shape = shape[None, :]
            if shape.shape[1] > 10:
                shape = shape[:, :10]

            # 3D joints
            trans = trans_cam[idxs].astype(np.float32)[sl]
            if set_name == 'train':
                # Compute 49 joints in camera coords; subtract root (index 39 in SPIN ordering)
                j3d = compute_smpl_joints49(pose, shape, trans, smpl_model=smpl_model)
                j3d = j3d - j3d[:, 39:40, :]
            else:
                # For val/test, compute 49 and map to COMMON-14 (evaluation-friendly); root-centered
                j3d_full = compute_smpl_joints49(pose, shape, trans, smpl_model=smpl_model)
                j3d_full = j3d_full - j3d_full[:, 39:40, :]
                j3d = convert_kps(j3d_full, src='spin', dst='common')

            # Video/frame ids
            num_frames = len(img_paths_array)
            seq_name = key  # already unique per video + subject
            frame_ids = []
            for p in img_paths_array:
                base = osp.basename(p)
                stem, _ = osp.splitext(base)
                digits = ''.join(ch for ch in stem if ch.isdigit())
                try:
                    frame_ids.append(int(digits))
                except Exception:
                    frame_ids.append(0)
            frame_ids = np.array(frame_ids)

            # Fill dataset dict
            dataset['vid_name'].append(np.array([seq_name] * num_frames))
            dataset['frame_id'].append(frame_ids)
            dataset['img_name'].append(img_paths_array)
            dataset['joints2D'].append(j2d_common)
            dataset['bbox'].append(bbox)
            dataset['pose'].append(pose)
            dataset['shape'].append(shape)
            dataset['joints3D'].append(j3d)

            # Extract SPIN features on cropped images
            features = extract_features(model, None, img_paths_array, bbox,
                                        kp_2d=j2d_common, debug=debug, dataset='3dpw', scale=1.3, batch_size=hmr_batch)
            dataset['features'].append(features)

            # Update per-scene progress
            try:
                scene_pbar.update(len(img_paths_array))
            except Exception:
                pass

        scene_pbar.close()

    # Concatenate across scenes/videos
    for k in dataset.keys():
        dataset[k] = np.concatenate(dataset[k]) if len(dataset[k]) > 0 else np.array([])
        if isinstance(dataset[k], np.ndarray):
            print(k, dataset[k].shape)

    # Filter by keypoint visibility (like 3DPW util)
    if dataset['joints2D'].size > 0:
        indices_to_use = np.where((dataset['joints2D'][:, :, 2] > VIS_THRESH).sum(-1) > MIN_KP)[0]
        for k in dataset.keys():
            dataset[k] = dataset[k][indices_to_use]

    return dataset


def read_single_npz(npz_path: str, images_dir: str, set_name: str, model, debug: bool = False, hmr_batch: int = 200, smpl_model: Optional[SMPL] = None) -> Dict[str, np.ndarray]:
    """Process a single BEDLAM .npz and return a TCMR-style dataset dict."""
    dataset = {
        'vid_name': [],
        'frame_id': [],
        'joints3D': [],
        'joints2D': [],
        'shape': [],
        'pose': [],
        'bbox': [],
        'img_name': [],
        'features': [],
    }

    data = np.load(npz_path, allow_pickle=True)
    imgnames = [_to_str(x) for x in _ensure_list(data['imgname'])]
    gtkps = np.asarray(data['gtkps'])
    pose_cam = np.asarray(data['pose_cam'])
    betas = np.asarray(data['shape'])
    trans_cam = np.asarray(data['trans_cam'])
    subs = _ensure_list(data['sub']) if 'sub' in data.files else [''] * len(imgnames)

    groups = group_indices_by_video(imgnames, subs=subs)
    # Per-scene progress bar (by frames)
    total_frames = sum(len(v) for v in groups.values())
    scene_name = osp.splitext(osp.basename(npz_path))[0]
    scene_pbar = tqdm(total=total_frames, desc=f'scene {scene_name}', leave=False)
    for key, idxs in groups.items():
        idxs = sorted(idxs, key=lambda i: imgnames[i])
        img_paths = [resolve_image_path(images_dir, npz_path, imgnames[i]) for i in idxs]

        j2d_smpl = gtkps[idxs].astype(np.float32)
        if j2d_smpl.shape[-1] == 2:
            vis = np.ones((j2d_smpl.shape[0], j2d_smpl.shape[1], 1), dtype=np.float32)
            j2d_smpl = np.concatenate([j2d_smpl, vis], axis=-1)

        bbox_params, time_pt1, time_pt2 = get_smooth_bbox_params(j2d_smpl, vis_thresh=0.1, sigma=8)
        c_x, c_y, scale = bbox_params[:, 0], bbox_params[:, 1], bbox_params[:, 2]
        w = h = 150.0 / scale
        w = h = h * 1.1
        bbox = np.vstack([c_x, c_y, w, h]).T

        j2d_common = np.stack([smpl_2d_to_common14(kp) for kp in j2d_smpl], axis=0)

        sl = slice(time_pt1, time_pt2)
        img_paths_array = np.array(img_paths)[sl]
        j2d_common = j2d_common[sl]
        bbox = bbox[sl]

        pose = pose_cam[idxs].astype(np.float32)[sl]
        shape = betas[idxs].astype(np.float32)[sl]
        if shape.ndim == 1:
            shape = shape[None, :]
        if shape.shape[1] > 10:
            shape = shape[:, :10]

        trans = trans_cam[idxs].astype(np.float32)[sl]
        if set_name == 'train':
            j3d = compute_smpl_joints49(pose, shape, trans, smpl_model=smpl_model)
            j3d = j3d - j3d[:, 39:40, :]
        else:
            j3d_full = compute_smpl_joints49(pose, shape, trans, smpl_model=smpl_model)
            j3d_full = j3d_full - j3d_full[:, 39:40, :]
            j3d = convert_kps(j3d_full, src='spin', dst='common')

        num_frames = len(img_paths_array)
        seq_name = key
        frame_ids = []
        for p in img_paths_array:
            base = osp.basename(p)
            stem, _ = osp.splitext(base)
            digits = ''.join(ch for ch in stem if ch.isdigit())
            try:
                frame_ids.append(int(digits))
            except Exception:
                frame_ids.append(0)
        frame_ids = np.array(frame_ids)

        dataset['vid_name'].append(np.array([seq_name] * num_frames))
        dataset['frame_id'].append(frame_ids)
        dataset['img_name'].append(img_paths_array)
        dataset['joints2D'].append(j2d_common)
        dataset['bbox'].append(bbox)
        dataset['pose'].append(pose)
        dataset['shape'].append(shape)
        dataset['joints3D'].append(j3d)

        features = extract_features(model, None, img_paths_array, bbox,
                                    kp_2d=j2d_common, debug=debug, dataset='3dpw', scale=1.3, batch_size=hmr_batch)
        dataset['features'].append(features)
        try:
            scene_pbar.update(len(img_paths_array))
        except Exception:
            pass

    for k in dataset.keys():
        dataset[k] = np.concatenate(dataset[k]) if len(dataset[k]) > 0 else np.array([])

    if dataset['joints2D'].size > 0:
        indices_to_use = np.where((dataset['joints2D'][:, :, 2] > VIS_THRESH).sum(-1) > MIN_KP)[0]
        for k in dataset.keys():
            dataset[k] = dataset[k][indices_to_use]

    scene_pbar.close()
    return dataset


def main():
    parser = argparse.ArgumentParser(description='Convert BEDLAM .npz to TCMR DB format')
    parser.add_argument('--npz_dir', type=str, default='E:/BEDLAM_npz', help='Folder with BEDLAM .npz files')
    parser.add_argument('--images_dir', type=str, default='E:/BEDLAM_image', help='BEDLAM images root folder')
    parser.add_argument('--set', type=str, default='train', choices=['train', 'val', 'test'])
    parser.add_argument('--split_file', type=str, default='', help='Optional text file with scene basenames to include')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--hmr_batch', type=int, default=200, help='Batch size for HMR feature extraction')
    parser.add_argument('--per_scene', action='store_true', help='Process and save per .npz scene to avoid OOM')
    parser.add_argument('--merge', action='store_true', help='Merge per-scene .pt files into a single set file')
    parser.add_argument('--merge_first_n', type=int, default=0,
                        help='Only take the first N frames from each per-scene .pt when merging (0 = all)')
    parser.add_argument('--sets', type=str, default='', help='Comma-separated sets to merge (default uses --set)')
    parser.add_argument('--overwrite', action='store_true', help='Recompute and overwrite existing .pt files')
    args = parser.parse_args()

    split_path = args.split_file if args.split_file else None
    os.makedirs(TCMR_DB_DIR, exist_ok=True)

    if args.merge:
        # Merge per-scene outputs into single file(s)
        sets = [s.strip() for s in args.sets.split(',') if s.strip()] if args.sets else [args.set]

        def list_scene_files(db_dir: str, set_name: str) -> List[str]:
            files = []
            for nm in os.listdir(db_dir):
                if not nm.startswith('bedlam_'):
                    continue
                if nm == f'bedlam_{set_name}_db_small.pt':
                    continue
                if nm.endswith(f'_{set_name}_db.pt'):
                    files.append(osp.join(db_dir, nm))
            return sorted(files)

        for s in sets:
            # If merged output already exists and no overwrite requested, skip
            merged_out_path = osp.join(TCMR_DB_DIR, f'bedlam_{s}_db_small.pt')
            if osp.exists(merged_out_path) and not args.overwrite:
                print(f'Skipping merge for set={s} because {merged_out_path} already exists. Use --overwrite to rebuild.')
                continue
            scene_files = list_scene_files(TCMR_DB_DIR, s)
            if not scene_files:
                print(f'No per-scene files found for set={s} in {TCMR_DB_DIR}')
                continue
            print(f'Merging {len(scene_files)} files for set={s}')
            merged = {k: [] for k in ['vid_name','frame_id','img_name','joints2D','bbox','pose','shape','joints3D','features']}
            for fpath in tqdm(scene_files, desc=f'Merge {s}'):
                part = joblib.load(fpath)

                # Determine how many frames to take from this scene
                take_n = None
                if args.merge_first_n and args.merge_first_n > 0:
                    # Compute the minimum available length across present keys to keep arrays aligned
                    lengths = []
                    for size_key in ['frame_id', 'img_name', 'joints2D', 'features', 'joints3D', 'pose', 'shape', 'bbox', 'vid_name']:
                        if size_key in part and hasattr(part[size_key], 'shape') and len(part[size_key]) > 0:
                            try:
                                lengths.append(int(part[size_key].shape[0]))
                            except Exception:
                                try:
                                    lengths.append(int(len(part[size_key])))
                                except Exception:
                                    pass
                    if lengths:
                        take_n = min(args.merge_first_n, min(lengths))

                for k in merged.keys():
                    if k in part and len(part[k]) > 0:
                        arr = part[k]
                        if take_n is not None:
                            arr = arr[:take_n]
                        merged[k].append(arr)
                del part
                gc.collect()
            # Memory-efficient concatenation using pre-allocation
            for k in merged.keys():
                if not merged[k]:
                    merged[k] = np.array([])
                    continue

                # For large arrays like features, use pre-allocation to avoid OOM
                if k == 'features' and len(merged[k]) > 1:
                    print(f"Pre-allocating and copying {k} to avoid OOM...")
                    try:
                        # Calculate total size needed
                        total_size = sum(arr.shape[0] for arr in merged[k])
                        feature_dim = merged[k][0].shape[1] if len(merged[k][0].shape) > 1 else 1

                        # Pre-allocate result array
                        if len(merged[k][0].shape) > 1:
                            result = np.empty((total_size, feature_dim), dtype=merged[k][0].dtype)
                        else:
                            result = np.empty((total_size,), dtype=merged[k][0].dtype)

                        # Copy chunks into pre-allocated array
                        start_idx = 0
                        for i, chunk in enumerate(merged[k]):
                            end_idx = start_idx + chunk.shape[0]
                            result[start_idx:end_idx] = chunk
                            start_idx = end_idx
                            # Free memory of processed chunk
                            merged[k][i] = None
                            if i % 2 == 0:  # Collect garbage every 2 chunks
                                gc.collect()
                        merged[k] = result
                    except MemoryError:
                        print(f"Still OOM with pre-allocation. Trying chunked processing for {k}...")
                        # Fallback: save features in smaller files and merge later
                        chunk_size = max(1, len(merged[k]) // 4)  # Process in quarters
                        chunks = []
                        for i in range(0, len(merged[k]), chunk_size):
                            chunk_data = merged[k][i:i+chunk_size]
                            if chunk_data:
                                chunk_result = np.concatenate(chunk_data)
                                chunks.append(chunk_result)
                                # Clear processed data
                                for j in range(i, min(i+chunk_size, len(merged[k]))):
                                    merged[k][j] = None
                                gc.collect()
                        merged[k] = np.concatenate(chunks) if chunks else np.array([])
                else:
                    merged[k] = np.concatenate(merged[k])

                if isinstance(merged[k], np.ndarray):
                    print(s, k, merged[k].shape)

            # Optional visibility filter
            if merged['joints2D'].size > 0:
                indices_to_use = np.where((merged['joints2D'][:, :, 2] > VIS_THRESH).sum(-1) > MIN_KP)[0]
                for k in merged.keys():
                    merged[k] = merged[k][indices_to_use]

            out_name = f'bedlam_{s}_db.pt'
            joblib.dump(merged, osp.join(TCMR_DB_DIR, out_name))
            print('Saved merged DB to', osp.join(TCMR_DB_DIR, out_name))

    elif args.per_scene:
        # Prepare list of .npz files based on split
        all_npz = [osp.join(args.npz_dir, f) for f in os.listdir(args.npz_dir) if f.endswith('.npz')]
        if split_path is not None and osp.isfile(split_path):
            with open(split_path, 'r') as f:
                allowed = {line.strip() for line in f if line.strip()}
            all_npz = [p for p in all_npz if osp.splitext(osp.basename(p))[0] in allowed]

        model = spin.get_pretrained_hmr()
        smpl_model = SMPL(SMPL_MODEL_DIR, batch_size=1, create_transl=False)
        for npz_path in tqdm(sorted(all_npz), desc=f'BEDLAM per-scene {args.set}'):
            scene = osp.splitext(osp.basename(npz_path))[0]
            out_name = f'bedlam_{scene}_{args.set}_db.pt'
            out_path = osp.join(TCMR_DB_DIR, out_name)
            if osp.exists(out_path) and not args.overwrite:
                print(f'Skipping scene {scene} because {out_path} already exists. Use --overwrite to rebuild.')
                continue
            dataset = read_single_npz(npz_path, args.images_dir, args.set, model, debug=args.debug, hmr_batch=args.hmr_batch, smpl_model=smpl_model)
            joblib.dump(dataset, out_path)
            print('Saved DB to', out_path)
            # best-effort memory cleanup
            del dataset
            gc.collect()
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
    else:
        out_name = f'bedlam_{args.set}_db.pt'
        out_path = osp.join(TCMR_DB_DIR, out_name)
        if osp.exists(out_path) and not args.overwrite:
            print(f'Skipping full-set build because {out_path} already exists. Use --overwrite to rebuild.')
            return
        dataset = read_data(args.npz_dir, args.images_dir, args.set, split_file=split_path, debug=args.debug, hmr_batch=args.hmr_batch)
        joblib.dump(dataset, out_path)
        print('Saved DB to', out_path)


if __name__ == '__main__':
    main()
