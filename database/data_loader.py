from database.iSeg.data_loader import  Loader as DL_iSeg
from database.WMH.data_loader import Loader as DL_WMH
from database.IBSR.data_loader import Loader as DL_IBSR
from database.BraTS.data_loader import Loader as DL_BRATS
from src.config import DB

class Loader():

    def __init__(self, data_path):
        self.data_path = data_path

    @staticmethod
    def create(db_name):
        if db_name == 'iSeg':
            return DL_iSeg(DB.iSEG['data_dir'])
        elif db_name == 'WMH':
            return DL_WMH(DB.WMH['data_dir'])
        elif db_name == 'LPBA40':
            return DL_IBSR(DB.IBSR['data_dir'])
        elif db_name == 'BraTS2017':
            return DL_BRATS(DB.BRATS2017['data_dir'])
        else:
            raise ValueError('Please, specify a valid DB  name')

