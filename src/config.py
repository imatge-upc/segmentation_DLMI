'''CONFIGURATION FILE'''

'''Database definition'''
class DB():
    BRATS2016 = {
            'db_name': 'BraTS2016',
            'data_dir': ['/projects/neuro/BRATS/BRATS2015_Training/HGG',
                      '/projects/neuro/BRATS/BRATS2015_Training/LGG'],
            }

    BRATS2017 = {
        'db_name': 'BraTS2017',
        'data_dir': ['/projects/neuro/BRATS/BRATS2017_Training/HGG',
                     '/projects/neuro/BRATS/BRATS2017_Training/LGG'],
    }

    BRATS2017_Validation = {
        'db_name': 'BRATS2017_Validation',
        'data_dir': '/projects/neuro/BRATS/BRATS2017_Validation'
    }

    BRATS2017_Test = {
        'db_name': 'BRATS2017_Test',
        'data_dir': '/projects/neuro/BRATS/BRATS2017_Testing'
    }

    WMH = {
        'db_name': 'WMH',
        'data_dir': ['/projects/neuro/WMH/Utrecht','/projects/neuro/WMH/GE3T', '/projects/neuro/WMH/Singapore']
    }

    iSEG = {
        'db_name': 'iSeg',
        'data_dir': ['/projects/neuro/iSeg-2017/Training']
    }

    iSEG_test = {
        'db_name': 'iSeg',
        'data_dir': ['/projects/neuro/iSeg-2017/Testing']
    }

    IBSR = {
        'db_name': 'IBSR',
        'data_dir': ['/projects/neuro/IBSR']
    }