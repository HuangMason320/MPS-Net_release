from lib.dataset import Dataset3D
from lib.core.config import EMDB_DIR

class EMDB(Dataset3D):
    def __init__(self, load_opt, set, seqlen, overlap=0.75, debug=False, target_vid=''):
        db_name = 'emdb'

        print('EMDB Dataset overlap ratio: ', overlap)
        super(EMDB, self).__init__(
            load_opt=load_opt,
            set=set,
            folder=EMDB_DIR,
            seqlen=seqlen,
            overlap=overlap,
            dataset_name=db_name,
            debug=debug,
            target_vid=target_vid
        )
        print(f'{db_name} - number of dataset objects {self.__len__()}')