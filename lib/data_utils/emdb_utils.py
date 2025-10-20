import os
import os.path as osp
import argparse
import pickle as pkl
from tqdm import tqdm

import numpy as np
import joblib
import torch

from lib.models import spin
from lib.core.config import TCMR_DB_DIR, EMDB_DIR, BASE_DATA_DIR
from lib.models.smpl import SMPL, SMPL_MODEL_DIR, H36M_TO_J14
from lib.utils.smooth_bbox import get_smooth_bbox_params
from lib.data_utils._feature_extractor import extract_features


VIS_THRESH = 0.3
MIN_KP = 6


def _compose_smpl_params(poses_root, poses_body, betas, trans):
    """Compose per-frame SMPL parameters.

    poses_root: (T,3) axis-angle global orient
    poses_body: (T,69) axis-angle body pose
    betas: (10,) or (T,10)
    trans: (T,3)
    """
    T = poses_root.shape[0]
    pose = np.zeros((T, 72), dtype=np.float32)
    pose[:, :3] = poses_root.astype(np.float32)
    pose[:, 3:] = poses_body.astype(np.float32)

    if betas.ndim == 1:
        betas = np.tile(betas[None, :], (T, 1))
    else:
        betas = betas.astype(np.float32)

    trans = trans.astype(np.float32)
    return pose, betas, trans


def _smpl_to_common_kp2d(kp2d_smpl):
    """Map SMPL 2D joints (24x2) to 'common' format (14x3).

    The 'common' order is
    [rankle, rknee, rhip, lhip, lknee, lankle, rwrist, relbow, rshoulder, lshoulder, lelbow, lwrist, neck, headtop].

    We fill the visibility channel; neck and headtop are set to invisible similar to 3DPW preprocessing.
    """
    # kp2d_smpl: (T,24,2) or (24,2)
    single = False
    if kp2d_smpl.ndim == 2:
        kp2d_smpl = kp2d_smpl[None]
        single = True

    T = kp2d_smpl.shape[0]
    j2d = np.zeros((T, 14, 3), dtype=np.float32)

    # Indices in SMPL joint order (see get_smpl_joint_names in _kp_utils.py)
    SMPL = {
        'hips': 0,
        'leftUpLeg': 1,
        'rightUpLeg': 2,
        'spine': 3,
        'leftLeg': 4,
        'rightLeg': 5,
        'spine1': 6,
        'leftFoot': 7,
        'rightFoot': 8,
        'spine2': 9,
        'leftToeBase': 10,
        'rightToeBase': 11,
        'neck': 12,
        'leftShoulder': 13,
        'rightShoulder': 14,
        'head': 15,
        'leftArm': 16,
        'rightArm': 17,
        'leftForeArm': 18,
        'rightForeArm': 19,
        'leftHand': 20,
        'rightHand': 21,
        'leftHandIndex1': 22,
        'rightHandIndex1': 23,
    }

    # common indices mapping: (dst index) -> (src smpl index)
    mapping = {
        0: SMPL['rightFoot'],     # rankle
        1: SMPL['rightLeg'],      # rknee
        2: SMPL['rightUpLeg'],    # rhip
        3: SMPL['leftUpLeg'],     # lhip
        4: SMPL['leftLeg'],       # lknee
        5: SMPL['leftFoot'],      # lankle
        6: SMPL['rightHand'],     # rwrist
        7: SMPL['rightForeArm'],  # relbow
        8: SMPL['rightShoulder'], # rshoulder
        9: SMPL['leftShoulder'],  # lshoulder
        10: SMPL['leftForeArm'],  # lelbow
        11: SMPL['leftHand'],     # lwrist
        12: SMPL['neck'],         # neck (set invis later to match 3DPW style)
        # 13 -> headtop (not available in SMPL); will remain zeros
    }

    for dst_idx, smpl_idx in mapping.items():
        j2d[:, dst_idx, :2] = kp2d_smpl[:, smpl_idx, :2]
        j2d[:, dst_idx, 2] = 1.0

    # Set neck and headtop visibility to 0.0 as in 3DPW preprocessing
    j2d[:, 12:, 2] = 0.0

    if single:
        j2d = j2d[0]
    return j2d


def _empty_dataset_container():
    return {
        'vid_name': [],
        'frame_id': [],
        'joints3D': [],
        'joints2D': [],
        'shape': [],
        'pose': [],
        'bbox': [],
        'img_name': [],
        'features': [],
        'valid': [],
    }


def _finalize_dataset(dataset):
    for k in dataset.keys():
        dataset[k] = np.concatenate(dataset[k]) if len(dataset[k]) > 0 else np.array([])
        if isinstance(dataset[k], np.ndarray):
            print(k, dataset[k].shape)
    if dataset['joints2D'].size > 0:
        indices_to_use = np.where((dataset['joints2D'][:, :, 2] > VIS_THRESH).sum(-1) > MIN_KP)[0]
        for k in dataset.keys():
            dataset[k] = dataset[k][indices_to_use]
    return dataset


def _merge_datasets(datasets):
    if not datasets:
        return _finalize_dataset(_empty_dataset_container())
    merged = {k: [] for k in datasets[0].keys()}
    for ds in datasets:
        if ds and isinstance(ds, dict) and ds.get('vid_name', None) is not None:
            for k in merged.keys():
                merged[k].append(ds[k])
    for k in merged.keys():
        merged[k] = np.concatenate(merged[k]) if len(merged[k]) > 0 else np.array([])
    return merged


def _process_subject(folder, subj, set, model, smpl, J_regressor=None, debug=False):
    dataset = _empty_dataset_container()
    subj_dir = osp.join(folder, subj)
    if not osp.isdir(subj_dir):
        return _finalize_dataset(dataset)

    seqs = sorted([s for s in os.listdir(subj_dir) if osp.isdir(osp.join(subj_dir, s))])
    for seq in tqdm(seqs, desc=f'{subj}'):
        seq_dir = osp.join(subj_dir, seq)
        img_dir = osp.join(seq_dir, 'images')
        pkl_files = [f for f in os.listdir(seq_dir) if f.endswith('_data.pkl')]
        if len(pkl_files) == 0:
            continue
        data = pkl.load(open(osp.join(seq_dir, pkl_files[0]), 'rb'))

        n_frames = int(data['n_frames'])
        good_mask = data['good_frames_mask'].astype(bool)

        cam = data['camera']
        extrinsics = np.asarray(cam['extrinsics']).astype(np.float32)

        smpl_dict = data['smpl']
        poses_root = np.asarray(smpl_dict['poses_root']).astype(np.float32)
        poses_body = np.asarray(smpl_dict['poses_body']).astype(np.float32)
        trans = np.asarray(smpl_dict['trans']).astype(np.float32)
        betas = np.asarray(smpl_dict['betas']).astype(np.float32)

        kp2d_smpl = np.asarray(data['kp2d']).astype(np.float32)  # (T,24,2)

        pose, shape, transl = _compose_smpl_params(poses_root, poses_body, betas, trans)
        img_paths = [osp.join(img_dir, f"{i:05d}.jpg") for i in range(n_frames)]
        j2d = _smpl_to_common_kp2d(kp2d_smpl)

        with torch.no_grad():
            p_t = torch.from_numpy(pose).float()
            s_t = torch.from_numpy(shape).float()
            t_t = torch.from_numpy(transl).float()
            out = smpl(betas=s_t, body_pose=p_t[:, 3:], global_orient=p_t[:, :3], transl=t_t)
            verts = out.vertices
            j3d_world = out.joints

        R = torch.from_numpy(extrinsics[:, :3, :3]).float()
        t = torch.from_numpy(extrinsics[:, :3, 3]).float()
        j3d_cam = torch.bmm(j3d_world, R.transpose(1, 2)) + t[:, None, :]

        if set == 'test' or set == 'validation':
            J_regressor_batch = J_regressor[None, :].expand(verts.shape[0], -1, -1).to(verts.device)
            j_h36m = torch.matmul(J_regressor_batch, verts)
            j3d_cam = j_h36m[:, H36M_TO_J14, :]

        j3d_cam_np = j3d_cam.cpu().numpy()

        bbox_params, time_pt1, time_pt2 = get_smooth_bbox_params(j2d, vis_thresh=VIS_THRESH, sigma=8)
        c_x, c_y, scale = bbox_params[:, 0], bbox_params[:, 1], bbox_params[:, 2]
        w = h = 150.0 / scale
        w = h = h * 1.1
        bbox = np.vstack([c_x, c_y, w, h]).T

        sl = slice(time_pt1, time_pt2)
        img_paths_array = np.array(img_paths)[sl]

        dataset['vid_name'].append(np.array([f'{subj}_{seq}'] * (time_pt2 - time_pt1)))
        dataset['frame_id'].append(np.arange(0, n_frames)[sl])
        dataset['img_name'].append(img_paths_array)
        dataset['joints3D'].append(j3d_cam_np[sl])
        dataset['joints2D'].append(j2d[sl])
        dataset['shape'].append(shape[sl])
        dataset['pose'].append(pose[sl])
        dataset['bbox'].append(bbox)
        dataset['valid'].append(good_mask[sl].astype(np.float32))

        features = extract_features(model, None, img_paths_array, bbox, kp_2d=j2d[sl], debug=debug, dataset='emdb', scale=1.3)
        dataset['features'].append(features)

    dataset = _finalize_dataset(dataset)
    return dataset


def read_data(folder, set, debug=False):
    model = spin.get_pretrained_hmr()
    smpl = SMPL(SMPL_MODEL_DIR, batch_size=1, create_transl=False, num_betas=10)

    J_regressor = None
    if set == 'test' or set == 'validation':
        J_regressor = torch.from_numpy(np.load(osp.join(BASE_DATA_DIR, 'J_regressor_h36m.npy'))).float()

    participants = sorted([d for d in os.listdir(folder) if d.startswith('P')])
    if set == 'test':
        subjects = ['P3', 'P9']
    else:
        subjects = [p for p in participants if p not in ['P3', 'P9']]

    all_datasets = []
    for subj in subjects:
        ds_subj = _process_subject(folder, subj, set, model, smpl, J_regressor, debug)
        if ds_subj['vid_name'].size == 0:
            continue
        all_datasets.append(ds_subj)
    merged = _merge_datasets(all_datasets)
    return merged


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, help='dataset directory', default=EMDB_DIR)
    args = parser.parse_args()

    debug = False

    # Initialize heavy models once
    model = spin.get_pretrained_hmr()
    smpl = SMPL(SMPL_MODEL_DIR, batch_size=1, create_transl=False, num_betas=10)
    J_regressor = torch.from_numpy(np.load(osp.join(BASE_DATA_DIR, 'J_regressor_h36m.npy'))).float()

    # Discover participants under the EMDB root
    participants = sorted([d for d in os.listdir(args.dir) if d.startswith('P')])

    # Train split: all except P3, P9
    train_subjects = [p for p in participants if p not in ['P3', 'P9']]
    train_datasets = []
    for subj in train_subjects:
        ds_subj = _process_subject(args.dir, subj, 'train', model, smpl, None, debug)
        if ds_subj['vid_name'].size == 0:
            continue
        out_path = osp.join(TCMR_DB_DIR, f'emdb_train_{subj}_db.pt')
        print(f'Saving {subj} train split to {out_path}')
        joblib.dump(ds_subj, out_path)
        train_datasets.append(ds_subj)

    if len(train_datasets) > 0:
        dataset_train = _merge_datasets(train_datasets)
        joblib.dump(dataset_train, osp.join(TCMR_DB_DIR, 'emdb_train_db.pt'))
    else:
        print('No training subjects processed. Skipping train merge.')

    # Test split: P3 and P9
    test_subjects = [p for p in participants if p in ['P3', 'P9']]
    test_datasets = []
    for subj in test_subjects:
        ds_subj = _process_subject(args.dir, subj, 'test', model, smpl, J_regressor, debug)
        if ds_subj['vid_name'].size == 0:
            continue
        out_path = osp.join(TCMR_DB_DIR, f'emdb_test_{subj}_db.pt')
        print(f'Saving {subj} test split to {out_path}')
        joblib.dump(ds_subj, out_path)
        test_datasets.append(ds_subj)

    if len(test_datasets) > 0:
        dataset_test = _merge_datasets(test_datasets)
        joblib.dump(dataset_test, osp.join(TCMR_DB_DIR, 'emdb_test_db.pt'))
    else:
        print('No test subjects processed. Skipping test merge.')
