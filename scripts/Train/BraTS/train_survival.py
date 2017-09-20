import numpy as np
import os
from os.path import join
import nibabel as nib
import csv
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
import glob
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras.models import Sequential,Model
from keras.optimizers import Adam, SGD, RMSprop
from keras.regularizers import l1_l2
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
import scipy
import nibabel as nib



plt.switch_backend('Agg')

VALIDATION_PATH = '/work/acasamitjana/segmentation/BraTS/finale/20170815/LR_0.0005_v_net1_4_v_net2_v2/results_test_morphology'
TRAINING_PATH = '/work/mcata/BRATS/NET2_train'
BRATS_TRAIN_PATH = '/projects/neuro/BRATS/BRATS2017_Training/'
BRATS_TEST_PATH = '/projects/neuro/BRATS/BRATS2017_Validation/'
SURVIVAL_PATH = '/projects/neuro/BRATS/Brats17ValidationData/survival_evaluation.csv'
MASK_PATH = glob.glob('/projects/neuro/BRATS/BRATS2017_Training/*/')
FEATURES_PATH_VAL = '/work/acasamitjana/segmentation/BraTS/survival/val_features.csv'
FEATURES_PATH_TRAIN = '/work/acasamitjana/segmentation/BraTS/old/survival/train_features.csv'

class Subject_write(object):

    def __init__(self,id, prediction_path, imaging_path):
        self.id = id
        self.prediction_path = prediction_path
        self.survival, self.age = self._get_survival()
        self.imaging_path = imaging_path


    def get_prediction(self):
        proxy = nib.load(self.prediction_path)
        return np.asarray(proxy.dataobj)


    def get_image(self,modality):
        proxy = nib.load(join(self.imaging_path,modality+'.nii.gz'))
        return np.asarray(proxy.dataobj)

    def _get_survival(self):
        with open(SURVIVAL_PATH, newline='') as clinical_file:
            reader = csv.DictReader(clinical_file)
            for row in reader:
                if row['Brats17ID'] == self.id:
                    return float(row['Survival']), float(row['Age'])
        return None, None

    def _get_survival_test(self):
        with open(SURVIVAL_PATH, newline='') as clinical_file:
            reader = csv.DictReader(clinical_file)
            for row in reader:
                if row['Brats17ID'] == self.id:
                    return float(row['Age'])
        return None, None


class Subject_write_test(object):

    def __init__(self,id, prediction_path, imaging_path):
        self.id = id
        self.prediction_path = prediction_path
        self.age = self._get_survival()
        self.imaging_path = imaging_path


    def get_prediction(self):
        proxy = nib.load(self.prediction_path)
        return np.asarray(proxy.dataobj)


    def get_image(self,modality):
        proxy = nib.load(join(self.imaging_path,modality+'.nii.gz'))
        return np.asarray(proxy.dataobj)


    def _get_survival(self):
        with open(SURVIVAL_PATH, newline='') as clinical_file:
            reader = csv.DictReader(clinical_file)
            for row in reader:
                if row['Brats17ID'] == self.id:
                    return float(row['Age'])
        return None, None


class Subject(object):

    def __init__(self, id, TIV, perc_necrotic, perc_enhancing, perc_edema, perc_WT,perc_ET,perc_TC,
                 mean_T1c_necrotic, mean_T1c_edema, mean_T1c_enhancing,
                 mean_T1_necrotic, mean_T1_edema, mean_T1_enhancing,
                 mean_T2_necrotic, mean_T2_edema, mean_T2_enhancing,
                 mean_FLAIR_necrotic, mean_FLAIR_edema, mean_FLAIR_enhancing,
                 std_T1c_necrotic, std_T1c_edema, std_T1c_enhancing,
                 std_T1_necrotic, std_T1_edema, std_T1_enhancing,
                 std_T2_necrotic, std_T2_edema, std_T2_enhancing,
                 std_FLAIR_necrotic, std_FLAIR_edema, std_FLAIR_enhancing,
                 age, survival=None):
        self.id = id
        self.TIV = TIV
        self.perc_necrotic=perc_necrotic
        self.perc_enhancing=perc_enhancing
        self.perc_edema=perc_edema
        self.perc_WT=perc_WT
        self.perc_ET=perc_ET
        self.perc_TC=perc_TC
        self.mean_T1c_necrotic=mean_T1c_necrotic
        self.mean_T1c_edema=mean_T1c_edema
        self.mean_T1c_enhancing=mean_T1c_enhancing
        self.mean_T1_necrotic=mean_T1_necrotic
        self.mean_T1_edema=mean_T1_edema
        self.mean_T1_enhancing=mean_T1_enhancing
        self.mean_T2_necrotic=mean_T2_necrotic
        self.mean_T2_edema=mean_T2_edema
        self.mean_T2_enhancing=mean_T2_enhancing
        self.mean_FLAIR_necrotic=mean_FLAIR_necrotic
        self.mean_FLAIR_edema=mean_FLAIR_edema
        self.mean_FLAIR_enhancing=mean_FLAIR_enhancing
        self.std_T1c_necrotic=std_T1c_necrotic
        self.std_T1c_edema=std_T1c_edema
        self.std_T1c_enhancing=std_T1c_enhancing
        self.std_T1_necrotic=std_T1_necrotic
        self.std_T1_edema=std_T1_edema
        self.std_T1_enhancing=std_T1_enhancing
        self.std_T2_necrotic=std_T2_necrotic
        self.std_T2_edema=std_T2_edema
        self.std_T2_enhancing=std_T2_enhancing
        self.std_FLAIR_necrotic=std_FLAIR_necrotic
        self.std_FLAIR_edema=std_FLAIR_edema
        self.std_FLAIR_enhancing=std_FLAIR_enhancing
        self.age=age
        self.survival=survival

    def get_T1c_features(self):
        return [self.mean_T1c_enhancing, self.mean_T1c_edema, self.mean_T1c_necrotic, self.std_T1c_enhancing, self.std_T1c_edema, self.std_T1c_necrotic]

    def get_T1_features(self):
        return [self.mean_T1_enhancing, self.mean_T1_edema, self.mean_T1_necrotic, self.std_T1_enhancing, self.std_T1_edema, self.std_T1_necrotic]


    def get_T2_features(self):
        return [self.mean_T2_enhancing, self.mean_T2_edema, self.mean_T2_necrotic, self.std_T2_enhancing, self.std_T2_edema, self.std_T2_necrotic]

    def get_FLAIR_features(self):
        return [self.mean_FLAIR_enhancing, self.mean_FLAIR_edema, self.mean_FLAIR_necrotic, self.std_FLAIR_enhancing, self.std_FLAIR_edema, self.std_FLAIR_necrotic]


def write_features():
    print('Creating subject lists')
    subject_list_val = read_images_test(VALIDATION_PATH, BRATS_TEST_PATH)
    # subject_list_train = read_images(TRAINING_PATH, BRATS_TRAIN_PATH)

    subject_list_val_filtered = list(filter(lambda sbj: not isinstance(sbj.age,tuple), subject_list_val))
    # subject_list_train_filtered = list(filter(lambda sbj: sbj.survival is not None, subject_list_train))

    title = ['ID', 'TIV', 'Perc_necrotic', 'Perc_edema', 'Perc_enhancing', 'Perc_WT', 'Perc_TC', 'Perc_ET', 'Age',
             'mean_T1c_necrotic','mean_T1c_edema', 'mean_T1c_enhancing',
             'mean_T1_necrotic', 'mean_T1_edema', 'mean_T1_enhancing',
             'mean_T2_necrotic','mean_T2_edema', 'mean_T2_enhancing',
             'mean_FLAIR_necrotic', 'mean_FLAIR_edema', 'mean_FLAIR_enhancing',
             'std_T1c_necrotic', 'std_T1c_edema', 'std_T1c_enhancing',
             'std_T1_necrotic', 'std_T1_edema', 'std_T1_enhancing',
             'std_T2_necrotic', 'std_T2_edema', 'std_T2_enhancing',
             'std_FLAIR_necrotic', 'std_FLAIR_edema', 'std_FLAIR_enhancing',
             ]

    path = FEATURES_PATH_VAL
    to_write = []
    for it_subject, subject in enumerate(subject_list_val_filtered):
        print('VAL: ' + str(it_subject))
        to_write.append(extract_features(subject,write=True))

    with open(path, 'a', newline='') as f:
        csv_writer = csv.writer(f, delimiter=',')
        csv_writer.writerow(title)
        csv_writer.writerows(to_write)

    # path = FEATURES_PATH_TRAIN
    # to_write = []
    # for it_subject, subject in enumerate(subject_list_train_filtered):
    #     print('TRAIN: ' + str(it_subject))
    #
    #     to_write.append(extract_features(subject,write=True))
    #
    # with open(path, 'a', newline='') as f:
    #     csv_writer = csv.writer(f, delimiter=',')
    #     csv_writer.writerow(title)
    #     csv_writer.writerows(to_write)


def extract_features(subject, write=False):
    array_prediction = subject.get_prediction()

    array_mask = subject.get_image('ROImask')
    T1 = subject.get_image('T1')
    T1c = subject.get_image('T1c')
    T2 = subject.get_image('T2')
    FLAIR = subject.get_image('FLAIR')

    voxels_brain = np.sum(array_mask)

    indices_necrotic = np.where(array_prediction == 1)
    indices_edema = np.where(array_prediction == 2)
    indices_enhancing = np.where(array_prediction == 4)

    voxels_necrotic = len(indices_necrotic[0])
    voxels_edema = len(indices_edema[0])
    voxels_enhancing = len(indices_enhancing[0])

    # survival, age = subject.survival, subject.age
    age = subject.age



    mean_T1c_necrotic = np.sum(T1c[indices_necrotic])/voxels_necrotic
    mean_T1c_edema = np.sum(T1c[indices_edema])/voxels_edema
    if voxels_enhancing == 0:
        mean_T1c_enhancing = 0
    else:
        mean_T1c_enhancing = np.sum(T1c[indices_enhancing])/voxels_enhancing

    mean_T1_necrotic = np.sum(T1[indices_necrotic])/voxels_necrotic
    mean_T1_edema = np.sum(T1[indices_edema])/voxels_edema
    if voxels_enhancing == 0:
        mean_T1_enhancing = 0
    else:
        mean_T1_enhancing = np.sum(T1[indices_enhancing])/voxels_enhancing

    mean_T2_necrotic = np.sum(T2[indices_necrotic])/voxels_necrotic
    mean_T2_edema = np.sum(T2[indices_edema])/voxels_edema
    if voxels_enhancing == 0:
        mean_T2_enhancing = 0
    else:
        mean_T2_enhancing = np.sum(T2[indices_enhancing])/voxels_enhancing

    mean_FLAIR_necrotic = np.sum(FLAIR[indices_necrotic])/voxels_necrotic
    mean_FLAIR_edema = np.sum(FLAIR[indices_edema])/voxels_edema
    if voxels_enhancing == 0:
        mean_FLAIR_enhancing = 0
    else:
        mean_FLAIR_enhancing = np.sum(FLAIR[indices_enhancing])/voxels_enhancing


    std_T1c_necrotic = np.sqrt((np.sum((T1c[indices_necrotic]-mean_T1c_necrotic)**2))/(voxels_necrotic-1))
    std_T1c_edema = np.sqrt((np.sum((T1c[indices_edema]-mean_T1c_edema)**2))/voxels_edema)
    if voxels_enhancing == 0:
        std_T1c_enhancing = 0
    else:
        std_T1c_enhancing = np.sqrt((np.sum((T1c[indices_enhancing]-mean_T1c_enhancing)**2))/voxels_enhancing)

    std_T1_necrotic = np.sqrt((np.sum((T1[indices_necrotic]-mean_T1_necrotic)**2))/voxels_necrotic)
    std_T1_edema = np.sqrt((np.sum((T1[indices_edema]-mean_T1_edema)**2))/voxels_edema)
    if voxels_enhancing == 0:
        std_T1_enhancing = 0
    else:
        std_T1_enhancing = np.sqrt((np.sum((T1[indices_enhancing]-mean_T1_enhancing)**2))/voxels_enhancing)

    std_T2_necrotic = np.sqrt((np.sum((T2[indices_necrotic]-mean_T2_necrotic)**2))/voxels_necrotic)
    std_T2_edema = np.sqrt((np.sum((T2[indices_edema]-mean_T2_edema)**2))/voxels_edema)
    if voxels_enhancing == 0:
        std_T2_enhancing = 0
    else:
        std_T2_enhancing = np.sqrt((np.sum((T2[indices_enhancing]-mean_T2_enhancing)**2))/voxels_enhancing)

    std_FLAIR_necrotic = np.sqrt((np.sum((FLAIR[indices_necrotic]-mean_FLAIR_necrotic)**2))/voxels_necrotic)
    std_FLAIR_edema = np.sqrt((np.sum((FLAIR[indices_edema]-mean_FLAIR_edema)**2))/voxels_edema)
    if voxels_enhancing == 0:
        std_FLAIR_enhancing = 0
    else:
        std_FLAIR_enhancing =np.sqrt((np.sum((FLAIR[indices_enhancing]-mean_FLAIR_enhancing)**2))/voxels_enhancing)

    if write:
        return [subject.id,
                voxels_brain,
                voxels_necrotic/voxels_brain,
                voxels_edema/voxels_brain,
                voxels_enhancing/voxels_brain,
                (voxels_enhancing+voxels_edema+voxels_necrotic)/voxels_brain,
                (voxels_enhancing+voxels_necrotic)/voxels_brain,
                voxels_enhancing/voxels_brain,
                age,
                mean_T1c_necrotic,
                mean_T1c_edema,
                mean_T1c_enhancing,
                mean_T1_necrotic,
                mean_T1_edema,
                mean_T1_enhancing,
                mean_T2_necrotic,
                mean_T2_edema,
                mean_T2_enhancing,
                mean_FLAIR_necrotic,
                mean_FLAIR_edema,
                mean_FLAIR_enhancing,
                std_T1c_necrotic,
                std_T1c_edema,
                std_T1c_enhancing,
                std_T1_necrotic,
                std_T1_edema,
                std_T1_enhancing,
                std_T2_necrotic,
                std_T2_edema,
                std_T2_enhancing,
                std_FLAIR_necrotic,
                std_FLAIR_edema,
                std_FLAIR_enhancing,
                ]
                # survival],

    else:
        return np.asarray([voxels_brain, voxels_necrotic, voxels_edema, voxels_enhancing, age])
        # return np.asarray([voxels_brain,voxels_necrotic, voxels_edema, voxels_enhancing, age])


def read_images(path, brats_path):

    subject_list = []
    for file in os.listdir(path):
        id = file.split('_')
        id_sbj = id[1] + '_' + id[2] + '_' + id[3] + '_1'
        print(id_sbj)
        subject_list.append(Subject_write(id_sbj,join(path,file), join(brats_path,id[0],id_sbj)))

    return subject_list

def read_images_test(path, brats_path):

    subject_list = []
    for file in os.listdir(path):
        id = file.split('_')
        id_sbj = id[0] + '_' + id[1] + '_' + id[2] + '_1'
        print(id_sbj)
        subject_list.append(Subject_write_test(id_sbj,join(path,file), join(brats_path,id_sbj)))

    return subject_list

def mean_squared_error_normalized(y_true,y_pred):
    import keras.backend as K

    return K.mean(K.square(y_pred - y_true), axis=-1)

    # return K.mean(K.square(y_pred - y_true)/y_true, axis=-1)

# return K.mean(tf.divide(K.square(y_pred - y_true),K.var(y_true,axis=-1,keepdims=True)), axis=-1)

if __name__ == "__main__":

    # write_features()
    # exit()

    subject_val = []
    with open(FEATURES_PATH_VAL, newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            subject_val.append(Subject(
                id = row['ID'],
                TIV = float(row['TIV']),
                perc_necrotic = float(row['Perc_necrotic']),
                perc_enhancing = float(row['Perc_enhancing']),
                perc_edema = float(row['Perc_edema']),
                perc_WT = float(row['Perc_WT']),
                perc_ET = float(row['Perc_ET']),
                perc_TC = float(row['Perc_TC']),
                mean_T1c_necrotic =  float(row['mean_T1c_necrotic']),
                mean_T1c_edema =  float(row['mean_T1c_edema']),
                mean_T1c_enhancing =  float(row['mean_T1c_enhancing']),
                mean_T1_necrotic =  float(row['mean_T1_necrotic']),
                mean_T1_edema =  float(row['mean_T1_edema']),
                mean_T1_enhancing =  float(row['mean_T1_enhancing']),
                mean_T2_necrotic =  float(row['mean_T2_necrotic']),
                mean_T2_edema =  float(row['mean_T2_edema']),
                mean_T2_enhancing =  float(row['mean_T2_enhancing']),
                mean_FLAIR_necrotic =  float(row['mean_FLAIR_necrotic']),
                mean_FLAIR_edema =  float(row['mean_FLAIR_edema']),
                mean_FLAIR_enhancing =  float(row['mean_FLAIR_enhancing']),
                std_T1c_necrotic =  float(row['std_T1c_necrotic']),
                std_T1c_edema =  float(row['std_T1c_edema']),
                std_T1c_enhancing =  float(row['std_T1c_enhancing']),
                std_T1_necrotic =  float(row['std_T1_necrotic']),
                std_T1_edema =  float(row['std_T1_edema']),
                std_T1_enhancing =  float(row['std_T1_enhancing']),
                std_T2_necrotic =  float(row['std_T2_necrotic']),
                std_T2_edema =  float(row['std_T2_edema']),
                std_T2_enhancing =  float(row['std_T2_enhancing']),
                std_FLAIR_necrotic =  float(row['std_FLAIR_necrotic']),
                std_FLAIR_edema =  float(row['std_FLAIR_edema']),
                std_FLAIR_enhancing =  float(row['std_FLAIR_enhancing']),
                age =  float(row['Age']),
                # survival =  float(row['Survival'])
            ))



    subject_train = []
    with open(FEATURES_PATH_TRAIN, newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            subject_train.append(Subject(
                id = row['ID'],
                TIV = float(row['TIV']),
                perc_necrotic = float(row['Perc_necrotic']),
                perc_enhancing = float(row['Perc_enhancing']),
                perc_edema = float(row['Perc_edema']),
                perc_WT = float(row['Perc_WT']),
                perc_ET = float(row['Perc_ET']),
                perc_TC = float(row['Perc_TC']),
                mean_T1c_necrotic =  float(row['mean_T1c_necrotic']),
                mean_T1c_edema =  float(row['mean_T1c_edema']),
                mean_T1c_enhancing =  float(row['mean_T1c_enhancing']),
                mean_T1_necrotic =  float(row['mean_T1_necrotic']),
                mean_T1_edema =  float(row['mean_T1_edema']),
                mean_T1_enhancing =  float(row['mean_T1_enhancing']),
                mean_T2_necrotic =  float(row['mean_T2_necrotic']),
                mean_T2_edema =  float(row['mean_T2_edema']),
                mean_T2_enhancing =  float(row['mean_T2_enhancing']),
                mean_FLAIR_necrotic =  float(row['mean_FLAIR_necrotic']),
                mean_FLAIR_edema =  float(row['mean_FLAIR_edema']),
                mean_FLAIR_enhancing =  float(row['mean_FLAIR_enhancing']),
                std_T1c_necrotic =  float(row['std_T1c_necrotic']),
                std_T1c_edema =  float(row['std_T1c_edema']),
                std_T1c_enhancing =  float(row['std_T1c_enhancing']),
                std_T1_necrotic =  float(row['std_T1_necrotic']),
                std_T1_edema =  float(row['std_T1_edema']),
                std_T1_enhancing =  float(row['std_T1_enhancing']),
                std_T2_necrotic =  float(row['std_T2_necrotic']),
                std_T2_edema =  float(row['std_T2_edema']),
                std_T2_enhancing =  float(row['std_T2_enhancing']),
                std_FLAIR_necrotic =  float(row['std_FLAIR_necrotic']),
                std_FLAIR_edema =  float(row['std_FLAIR_edema']),
                std_FLAIR_enhancing =  float(row['std_FLAIR_enhancing']),
                age =  float(row['Age']),
                survival =  float(row['Survival'])
            ))


    N_train = len(subject_train)
    N_val = len(subject_val)
    nfeat = 10

    array_train = np.zeros((N_train,nfeat))
    array_val = np.zeros((N_val, nfeat))
    labels_train = np.zeros(N_train)
    labels_val = np.zeros(N_val)

    for n,subject in enumerate(subject_train):
        array_train[n] = [subject.age, subject.perc_TC, subject.perc_WT, subject.perc_ET] + subject.get_FLAIR_features()
                         # subject.get_T1c_features() + subject.get_T1_features() + subject.get_T2_features() + \
                         # subject.get_FLAIR_features()
        labels_train[n] = subject.survival

    for n,subject in enumerate(subject_val):
        array_val[n] = [subject.age, subject.perc_TC, subject.perc_WT, subject.perc_ET] + subject.get_FLAIR_features()
                       # subject.get_T1c_features() + subject.get_T1_features() + subject.get_T2_features() +\
                       # subject.get_FLAIR_features()
        # labels_val[n] = subject.survival


    print(array_train.shape)
    mean_train = np.mean(array_train,axis=0,keepdims=True)
    std_train = np.std(array_train,axis=0,keepdims=True)
    mean_val = mean_train#np.mean(array_val,axis=0,keepdims=True)
    std_val = std_train#np.std(array_val,axis=0, keepdims=True)
    array_train = (array_train - mean_train)/ std_train
    array_val = (array_val -  mean_val)/ std_val

    mean_lab_train = np.mean(labels_train)
    std_lab_train = np.std(labels_train)
    mean_lab_val = mean_lab_train#np.mean(labels_val)
    std_lab_val = std_lab_train#np.std(labels_val)

    # labels_train = (labels_train - mean_lab_train)/std_lab_train
    # labels_val = (labels_val - mean_lab_val)/std_lab_val


    print('Training...')
    clf = Sequential()
    clf.add(Dense(50, input_shape=(nfeat,), kernel_regularizer=l1_l2(l1=0.001,l2=0.1)))
    # clf.add(BatchNormalization())
    clf.add(Activation('relu'))
    clf.add(Dropout(0.2))

    clf.add(Dense(25, kernel_regularizer=l1_l2(l1=0.001,l2=0.1)))
    # clf.add(BatchNormalization())
    clf.add(Activation('relu'))
    clf.add(Dropout(0.2))

    clf.add(Dense(25, kernel_regularizer=l1_l2(l1=0.001,l2=0.1)))
    # clf.add(BatchNormalization())
    clf.add(Activation('relu'))
    clf.add(Dropout(0.2))

    clf.add(Dense(10, kernel_regularizer=l1_l2(l1=0.001,l2=0.01)))
    # clf.add(BatchNormalization())
    clf.add(Activation('relu'))
    # clf.add(Dropout(0.1))

    clf.add(Dense(10, kernel_regularizer=l1_l2(l1=0.001, l2=0.01)))
    # clf.add(BatchNormalization())
    clf.add(Activation('relu'))
    # clf.add(Dropout(0.1))

    clf.add(Dense(1))

    lr = 0.01
    optimizer = Adam(lr=lr)#'Adam'#
    clf.compile(optimizer, loss=mean_squared_error_normalized)
    clf.summary()

    def schedule(epoch):
        return lr * (1-0.0005*epoch)
    callback = [LearningRateScheduler(schedule=schedule), ModelCheckpoint('/work/acasamitjana/segmentation/BraTS/survival/scatter_predictions.h5', save_best_only=True)]

    # clf.fit(array_train, labels_train, epochs=300, validation_data=(array_val, labels_val), callbacks=callback)
    # clf.save_weights('/work/acasamitjana/segmentation/BraTS/survival/scatter_predictions.png')

    # clf = Lasso()
    # clf.fit(array_train,labels_train)

    clf.load_weights('/work/acasamitjana/segmentation/BraTS/survival/scatter_predictions_v2.h5')
    predictions_train = clf.predict(array_train)
    predictions_val = clf.predict(array_val)

    TEST = True
    if TEST:
        with open('/work/acasamitjana/segmentation/BraTS/survival/val_predictions.csv', 'a', newline='') as f:
            csv_writer = csv.writer(f, delimiter=',')
            # csv_writer.writerow(['Brats17ID', 'Survival_prediction'])
            for i in range(len(subject_val)):
                csv_writer.writerow([subject_val[i].id,predictions_val[i][0]* std_lab_train + mean_lab_train])


    print(predictions_val)
    exit()
    print('Normalizing back features and labels')
    array_val = array_val * std_val+ mean_val
    labels_val = labels_val * mean_lab_val + mean_lab_val
    predictions_val = predictions_val * std_lab_val + mean_lab_val

    array_train = array_train * std_train + mean_train
    labels_train = labels_train * std_lab_train + mean_lab_train
    predictions_train = predictions_train * std_lab_train + mean_lab_train

    print('Report: ')
    print('TRAIN:')
    mse_train = 1/len(labels_train) * np.sum((labels_train - predictions_train) ** 2)
    predictions_label_train = np.ones_like(predictions_train)
    labels_label_train = np.ones_like(labels_train)
    predictions_label_train[predictions_train < 300] = 0
    predictions_label_train[predictions_train > 450] = 2
    labels_label_train[labels_train < 300] = 0
    labels_label_train[labels_train > 450] = 2
    print('MSE: ' + str(mse_train))
    print('Accuracy: ' + str(accuracy_score(labels_label_train, predictions_label_train)))


    print('VAL:')
    mse_val = 1/len(labels_val) * np.sum((labels_val - predictions_val) ** 2)
    predictions_label_val = np.ones_like(predictions_val)
    labels_label_val = np.ones_like(labels_val)
    predictions_label_val[predictions_val < 300] = 0
    predictions_label_val[predictions_val > 450] = 2
    labels_label_val[labels_val < 300] = 0
    labels_label_val[labels_val > 450] = 2
    print('MSE: ' + str(mse_val))
    print('Accuracy: ' + str(accuracy_score(labels_label_val, predictions_label_val)))



    print('Plotting predictions ...')
    print(labels_val.shape)
    print(predictions_val.shape)
    p = scipy.stats.pearsonr(np.squeeze(labels_train),np.squeeze(predictions_train))
    plt.figure()
    plt.plot(labels_train, predictions_train, '*b')
    plt.plot([300,300], [0,1000],'k')
    plt.plot([450,450], [0,1000],'k')
    plt.plot([0,1750], [300,300],'k')
    plt.plot([0,1750], [450,450],'k')

    plt.title('Pearson correlation: ' + str(p))
    plt.xlabel('Overall survival time [days]')
    plt.ylabel('Predicted Overall survival time [days]')
    plt.savefig('/work/acasamitjana/segmentation/BraTS/survival/scatter_predictions_train.png')
    plt.close()

    print('Plotting predictions ...')
    print(labels_val.shape)
    print(predictions_val.shape)
    p = scipy.stats.pearsonr(np.squeeze(labels_val),np.squeeze(predictions_val))
    plt.figure()
    plt.plot(labels_val, predictions_val, '*b')
    plt.plot([300,300], [0,1000],'k')
    plt.plot([450,450], [0,1000],'k')
    plt.plot([0,1750], [300,300],'k')
    plt.plot([0,1750], [450,450],'k')

    plt.title('Pearson correlation: ' + str(p))
    plt.xlabel('Overall survival time [days]')
    plt.ylabel('Predicted Overall survival time [days]')
    plt.savefig('/work/acasamitjana/segmentation/BraTS/survival/scatter_predictions_val.png')
    plt.close()

    print('Plotting WT ...')
    p = scipy.stats.pearsonr(np.squeeze(labels_train),np.squeeze(array_train[:,0]))
    plt.figure()
    plt.plot(labels_val, array_val[:,0], '*b')
    plt.title('Pearson correlation: ' + str(p))
    plt.xlabel('Overall survival time [days]')
    plt.ylabel('Normalized WT voxels')
    plt.savefig('/work/acasamitjana/segmentation/BraTS/survival/scatter_WT.png')

    plt.close()

    print('Plotting TC ...')
    p = scipy.stats.pearsonr(np.squeeze(labels_train),np.squeeze(array_train[:,1]))
    plt.figure()
    plt.plot(labels_val, array_val[:,1], '*b')
    plt.title('Pearson correlation: ' + str(p))
    plt.xlabel('Overall survival time [days]')
    plt.ylabel('Normalized TC voxels')
    plt.savefig('/work/acasamitjana/segmentation/BraTS/survival/scatter_TC.png')

    plt.close()

    print('Plotting ET ...')
    p = scipy.stats.pearsonr(np.squeeze(labels_train),np.squeeze(array_train[:,2]))
    plt.figure()
    plt.plot(labels_val, array_val[:,2], '*b')
    plt.title('Pearson correlation: ' + str(p))
    plt.xlabel('Overall survival time [days]')
    plt.ylabel('Normalized ET voxels')
    plt.savefig('/work/acasamitjana/segmentation/BraTS/survival/scatter_ET.png')

    plt.close()

    print('Plotting age ...')
    p = scipy.stats.pearsonr(np.squeeze(labels_train),np.squeeze(array_train[:,3]))
    plt.figure()
    plt.plot(labels_val, array_val[:,3], '*b')
    plt.title('Pearson correlation: ' + str(p))
    plt.xlabel('Overall survival time [days]')
    plt.ylabel('Age [years]')
    plt.savefig('/work/acasamitjana/segmentation/BraTS/survival/scatter_age.png')

    plt.close()




    # print('Plotting...')
    # index_order = np.argsort(labels_val)
    # plt.figure()
    # plt.plot(labels_val[index_order], 'b')
    # plt.plot(predictions_val[index_order], 'r')
    # plt.show()
    #
    # print('Plotting...')
    # index_order = np.argsort(labels_train)
    # plt.figure()
    # plt.plot(labels_train[index_order],'b')
    # plt.plot(predictions_train[index_order], 'r')
    # plt.show()






