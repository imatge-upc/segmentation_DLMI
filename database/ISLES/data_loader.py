import csv          # for the loader

import nibabel as nib
import numpy as np
import os
from os.path import join

import params as p

to_int = lambda b: 1 if b else 0

class Subject:
    def __init__(self,
                 data_path,
                 subject_name,
                 booleanFLAIR=True,
                 booleanT1=True,
                 booleanT1c=True,
                 booleanT2=True,
                 booleanROImask=True,
                 booleanLabels=True):

        self.data_path = data_path
        self.subject = subject_name
        self._id = os.path.basename(os.path.normpath(self.data_path)) + '_' + subject_name

        self.subject_filepath = join(self.data_path, self.subject)

        self.FLAIR_FILE = os.path.join(self.subject_filepath,'FLAIR.nii.gz') if booleanFLAIR is True else None
        self.T1_FILE = os.path.join(self.subject_filepath,'T1.nii.gz') if booleanT1 is True else None
        self.T1c_FILE = os.path.join(self.subject_filepath,'T1c.nii.gz') if booleanT1c is True else None
        self.T2_FILE = os.path.join(self.subject_filepath,'T2.nii.gz') if booleanT2 is True else None
        self.GT_FILE = os.path.join(self.subject_filepath,'GT.nii.gz') if booleanLabels is True else None
        self.ROIMASK_FILE = os.path.join(self.subject_filepath,'ROImask.nii.gz') if booleanROImask is True else None

        self.booleanFLAIR = booleanFLAIR
        self.booleanT1 = booleanT1
        self.booleanT1c = booleanT1c
        self.booleanT2 = booleanT2
        self.booleanROImask = booleanROImask
        self.booleanLabels = booleanLabels

        self.num_input_modalities = to_int(self.booleanFLAIR) + to_int(self.booleanT1) + to_int(self.booleanT1c) + to_int(
            self.booleanT2)


        self.train0_val1_both2 = 2

    @property
    def id(self):
        return self._id

    def get_affine(self):
        return nib.load(self.GT_FILE).affine

    def _get_filepath(self, functions_list):
        channels_path = ''
        for func in functions_list:
            path = filter(func, listdir(join(self.subject_filepath, channels_path)))
            if not path:
                raise ValueError(
                    'There is no matching of your search. Please check your database for subject ' + self.id)
            if len(path) > 1:
                raise ValueError('There are several files matching your search. Please be more specific in'
                                 'input channels. Please check your database for subject ' + self.id)
            path = path[0]
            channels_path = join(channels_path, path)
        return join(self.subject_filepath, channels_path)

    def load_channels(self):
        modalities = []
        modalities.append(nib.load(self.FLAIR_FILE)) if self.booleanFLAIR else None
        modalities.append(nib.load(self.T1_FILE)) if self.booleanT1 else None
        modalities.append(nib.load(self.T1c_FILE)) if self.booleanT1c else None
        modalities.append(nib.load(self.T2_FILE)) if self.booleanT2 else None


        channels = np.zeros((self.num_input_modalities,) + modalities[0].shape, dtype=np.float32)

        for index_mod, mod in enumerate(modalities):
            channels[index_mod] = np.asarray(mod.dataobj)

        return channels

    def load_labels(self):
        labels_proxy = nib.load(self.GT_FILE)
        labels = np.squeeze(np.asarray(labels_proxy.dataobj))
        return labels

    def load_ROI_mask(self):
        if self.booleanROImask:
            roimask_proxy = nib.load(self.ROIMASK_FILE)
            mask = np.asarray(roimask_proxy.dataobj)
            return mask.astype('bool')
        else:
            return None

    def get_subject_shape(self):

        if self.booleanLabels:
            proxy = nib.load(self.GT_FILE)
        elif self.booleanT1:
            proxy = nib.load(self.T1_FILE)
        elif self.booleanT1c:
            proxy = nib.load(self.T1c_FILE)
        elif self.booleanT2:
            proxy = nib.load(self.T2_FILE)
        elif self.booleanFLAIR:
            proxy = nib.load(self.FLAIR_FILE)
        else:
            raise ValueError("[src.helpers.Subject] No modalities are given for subject: " + self.id)

        return proxy.shape

class Subject_test:
    def __init__(self,
                 data_path,
                 subject_name,
                 booleanFLAIR=True,
                 booleanT1=True,
                 booleanT1c=True,
                 booleanT2=True):
        self.data_path = data_path
        self.subject = subject_name
        self._id = os.path.basename(os.path.normpath(self.data_path)) + '_' + subject_name

        self.subject_filepath = join(self.data_path, self.subject)
        self.FLAIR_FILE = join(self.subject_filepath, 'flair_corrected.nii.gz')
        self.T1_FILE = join(self.subject_filepath, 't1_corrected.nii.gz')
        self.T1c_FILE = join(self.subject_filepath, 't1c_corrected.nii.gz')
        self.T2_FILE = join(self.subject_filepath, 't2_corrected.nii.gz')
        self.ROIMASK_FILE = join(self.subject_filepath, 't1c_corrected.nii.gz')

        self.booleanFLAIR = booleanFLAIR
        self.booleanT1 = booleanT1
        self.booleanT1c = booleanT1c
        self.booleanT2 = booleanT2

    @property
    def id(self):
        return self._id

    def get_affine(self):
        return nib.load(self.T1c_FILE).affine

    def load_channels(self):
        flair = nib.load(self.FLAIR_FILE)
        t1 = nib.load(self.T1_FILE)
        t1c = nib.load(self.T1c_FILE)
        t2 = nib.load(self.T2_FILE)
        to_int = lambda b: 1 if b else 0
        num_input_modalities = to_int(self.booleanFLAIR) + to_int(self.booleanT1) + to_int(self.booleanT1c) + to_int(
            self.booleanT2)

        channels = np.zeros((num_input_modalities,) + flair.shape, dtype=np.float32)

        channels[0] = np.asarray(flair.dataobj) if self.booleanFLAIR is True else None
        channels[to_int(self.booleanFLAIR)] = np.asarray(t1.dataobj) if self.booleanT1 is True else None
        channels[to_int(self.booleanFLAIR) + to_int(self.booleanT1)] = np.asarray(
            t1c.dataobj) if self.booleanT1c is True else None
        channels[to_int(self.booleanFLAIR) + to_int(self.booleanT1) + to_int(self.booleanT1c)] = np.asarray(
            t2.dataobj) if self.booleanT2 is True else None

        return channels

    def load_ROI_mask(self):
        roimask_proxy = nib.load(self.ROIMASK_FILE)
        mask = np.asarray(roimask_proxy.dataobj)
        return mask.astype('bool')

class Loader():

    def __init__(self, data_path):
        self.data_path = data_path

    @staticmethod
    def create(config_dict):
        return Loader(config_dict['data_dir'])

    def load_subject(self, id):
        subject = Subject(id, self.data_path)

        # Search for subject in subjects
        found = False
        with open(self.subjects_file, newline='') as subjects_file:
            reader = csv.DictReader(subjects_file)
            for row in reader:
                if row['RID'] == id:
                    subject.age = self._get_value(row['AGE'])
                    subject.gender = self._get_value(row['GENDER'])
                    subject.school_years = self._get_value(row['EDUCATION'])
                    subject.apoe = self._get_value(row['APOE'])
                    subject.ab = self._get_value(row['AB'])
                    subject.ptau = self._get_value(row['PTAU'])
                    subject.ttau = self._get_value(row['TTAU'])
                    subject.diagnosis = self._get_value(row['DIAGNOSIS'])
                    found = True
                    break

        if not found:
            raise ValueError('The provided id ' + str(id) + ' is not found')

        return subject


    def load_subjects(self, parameters):
        booleanFLAIR = parameters[p.BOOLEAN_FLAIR]
        booleanT1 = parameters[p.BOOLEAN_T1]
        booleanT1c = parameters[p.BOOLEAN_T1C]
        booleanT2 = parameters[p.BOOLEAN_T2]
        booleanROImask = parameters[p.BOOLEAN_ROImask]
        booleanLabels = parameters[p.BOOLEAN_LABELS]

        subject_list = []
        data_path = self.data_path
        if not isinstance(self.data_path, list):
            data_path = [data_path]

        for path in data_path:
            for subject in os.listdir(path):
                subject_list.append(Subject(path,
                                            subject,
                                            booleanFLAIR=booleanFLAIR,
                                            booleanT1=booleanT1,
                                            booleanT1c=booleanT1c,
                                            booleanT2=booleanT2,
                                            booleanROImask=booleanROImask,
                                            booleanLabels=booleanLabels
                                            ))


    def _get_value(self, value):
        '''
         Checks that the value is not a nan, and returns None if it is
        '''
        if value == 'nan':
            return None

        if isinstance(value, str):
            # if value.isnumeric():
            #     return int(value)

            # Check if it can be casted to a float...
            try:
                return float(value)
            except ValueError:
                pass

        return value