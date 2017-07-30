from _ast import Lambda

from keras.layers import  Lambda, AveragePooling3D, MaxPooling3D, PReLU, Input, merge, multiply, Dense, Flatten,\
    Merge, BatchNormalization, Conv3D, Concatenate, Add, Activation, UpSampling3D, Multiply, GaussianNoise
from keras.models import Sequential, Model
from keras.optimizers import RMSprop, Adam, SGD, Adadelta
from keras.regularizers import l1_l2
from keras import backend as K
from src.layers import repeat_channels, repeat_channels_shape, complementary_mask, dice_1 as d1, dice_2 as d2, \
    dice_3 as d3, BatchNormalizationMasked
from src.activations import elementwise_softmax_3d
from src.losses import categorical_crossentropy_3d, scae_mean_squared_error_masked, dice_cost,\
    categorical_crossentropy_3d_masked,mean_squared_error_lambda, categorical_crossentropy_3d_lambda
from src.metrics import accuracy, dice_whole, dice_enhance, dice_core, recall_0,recall_1,recall_2,recall_3,\
    recall_4,precision_0,precision_1,precision_2,precision_3, precision_4, dice_0,dice_1,dice_2,dice_3
import h5py
import warnings
import tensorflow as tf


class SegmentationModels(object):
    """ Interface that allows you to save and load models and their weights """

    # ------------------------------------------------ CONSTANTS ------------------------------------------------ #

    DEFAULT_MODEL = 'v_net'

    # ------------------------------------------------- METHODS ------------------------------------------------- #

    @classmethod
    def get_model(cls, num_modalities, segment_dimensions, num_classes, model_name=None,
                  **kwargs):
        """
        Returns the compiled model specified by its name.
        If no name is given, the default model is returned, which corresponds to the
        hand-picked model that performs the best.

        Parameters
        ----------
        num_modalities : int
            Number of modalities used as input channels
        segment_dimensions : tuple
            Tuple with 3 elements that specify the shape of the input segments
        num_classes : int
            Number of output classes to be predicted in the segmentation
        model_name : [Optional] String
            Name of the model to be returned
        weights_filename: [Optional] String
            Path to the H5 file containing the weights of the trained model. The user must ensure the
            weights correspond to the model to be loaded.

        Returns
        -------
        keras.Model
            Compiled keras model
        tuple
            Tuple with output size (necessary to adjust the ground truth matrix's size),
            or None if output size is the sames as segment_dimensions.

        Raises
        ------
        TypeError
            If model_name is not a String or None
        ValueError
            If the model specified by model_name does not exist
        """
        if model_name is None:
            return cls.get_model(num_modalities, segment_dimensions, num_classes, model_name=cls.DEFAULT_MODEL,**kwargs)
        if isinstance(model_name, str):
            try:

                model_getter = cls.__dict__[model_name]
                return model_getter.__func__(num_modalities, segment_dimensions, num_classes,
                                             **kwargs)
            except KeyError:
                raise ValueError('Model {} does not exist. Use the class method {} to know the available model names'
                                 .format(model_name, 'get_model_names'))
        else:
            raise TypeError('model_name must be a String/Unicode sequence or None')

    @classmethod
    def get_model_names(cls):
        """
        List of available models' names

        Returns
        -------
        List
            Names of the available model names to be used in get_model(model_name) method
        """

        def filter_functions(attr_name):
            tmp = ('model' not in attr_name) and ('MODEL' not in attr_name)
            return tmp and ('__' not in attr_name) and ('BASE' not in attr_name)

        return filter(filter_functions, cls.__dict__.keys())

    @staticmethod
    def compile(model, lr=0.0005, num_classes = 2, loss_name = 'cross_entropy', optimizer_name = 'Adam'):
        if optimizer_name == 'Adam':
            beta_1 = 0.9
            beta_2 = 0.999
            epsilon = 10 ** (-8 )
            optimizer = Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, clipnorm=1.)
        elif optimizer_name == 'RMSProp':
            rho = 0.9
            epsilon =  10 ** (-8 )
            decay = 0.0
            optimizer = RMSprop(lr=lr, rho=rho, epsilon=epsilon, decay=decay)
        elif optimizer_name == 'SGD':
            optimizer = SGD(lr=lr, momentum=0.0, decay=0.0, nesterov=False)
        else:
            raise ValueError("Please, specify a valid optimizer when compiling")


        if num_classes == 2:
            metrics = [accuracy,dice_0,dice_1,recall_0,recall_1,precision_0,precision_1]
        elif num_classes == 3:
            metrics = [accuracy,dice_0,dice_1,dice_2,recall_0,recall_1,recall_2,precision_0,precision_1,precision_2]
        elif num_classes == 4:
            metrics = [accuracy,dice_0,dice_1,dice_2,dice_3,recall_0,recall_1,recall_2,recall_3,
                       precision_0,precision_1,precision_2,precision_3]
        elif num_classes == 5:
            metrics = [accuracy,dice_whole,dice_core,dice_enhance,recall_0,recall_1,recall_2,recall_3, recall_4,
                       precision_0,precision_1,precision_2,precision_3, precision_4]
        else:
            raise ValueError("Please, specify a valid number of classes when compiling")

        if loss_name == 'cross_entropy':
            print('Loss: cross_entropy')
            loss = categorical_crossentropy_3d
        elif loss_name == 'dice':
            print('Loss: dice')
            loss = dice_cost
        else:
            raise ValueError('Please, specify a valid loss function')

        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics

        )

        return model

    @staticmethod
    def compile_masked(model, lr=0.0005, num_classes=2):
        beta_1 = 0.9
        beta_2 = 0.999
        epsilon = 10 ** (-8)
        optimizer = Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, clipnorm=1.)

        loss = [lambda y_true, y_pred: y_pred]

        model.compile(
            optimizer=optimizer,
            loss=loss,

        )
        return model


    @staticmethod
    def two_pathways(num_modalities, segment_dimensions, num_classes):
        # TODO Document properly
        """
        TODO
        Returns
        -------
        keras.model
            Compiled Sequential model
        tuple (dim1, dim2, dim3)
            Output shape computed from segment_dimensions and the convolutional architecture
        """
        if not isinstance(segment_dimensions, tuple) or len(segment_dimensions) != 3:
            raise ValueError('segment_dimensions must be a tuple with length 3, specifying the shape of the '
                             'input segment')
        # for dim in segment_dimensions:
        #     assert dim % 32 == 0  # As there are 5 (2, 2, 2) max-poolings, 2 ** 5 is the minimum input size

        L1_reg = 0.000001
        L2_reg = 0.0001
        initializer = 'he_normal'

        # Compute input shape, receptive field and output shape after softmax activation
        input_shape = (num_modalities,) + segment_dimensions
        output_shape = segment_dimensions

        # Activations, regularizers and optimizers
        softmax_activation = Lambda(elementwise_softmax_3d, output_shape= (num_classes,) + segment_dimensions, name='softmax')
        regularizer = l1_l2(l1=L1_reg, l2=L2_reg)
        print(K.image_data_format())
        # Model
        x = Input(shape=input_shape, name='Two_pathways_input')

        # ------- Upper pathway ----
        tmp = Conv3D(8, (3,3,3), kernel_initializer=initializer, padding='same', activation='relu',
                            kernel_regularizer=regularizer, name='upper_conv_1')(x)
        tmp = BatchNormalization(axis=1, name='normalization1')(tmp)
        tmp = Conv3D(8, (3,3,3), kernel_initializer=initializer, padding='same', activation='relu',
                            kernel_regularizer=regularizer, name='upper_conv_2')(tmp)
        tmp = BatchNormalization(axis=1)(tmp)

        tmp = Conv3D(8, (3,3,3), kernel_initializer=initializer, padding='same', activation='relu',
                            kernel_regularizer=regularizer, name='upper_conv_3')(tmp)
        tmp = BatchNormalization(axis=1)(tmp)
        tmp = Conv3D(8, (3,3,3), kernel_initializer=initializer, padding='same', activation='relu',
                            kernel_regularizer=regularizer, name='upper_conv_4')(tmp)
        tmp = BatchNormalization(axis=1)(tmp)

        tmp = Conv3D(8, (3,3,3), kernel_initializer=initializer, padding='same', activation='relu',
                            kernel_regularizer=regularizer, name='upper_conv_5')(tmp)
        tmp = BatchNormalization(axis=1)(tmp)
        tmp = Conv3D(8, (3,3,3), kernel_initializer=initializer, padding='same', activation='relu',
                            kernel_regularizer=regularizer, name='upper_conv_6')(tmp)
        tmp = BatchNormalization(axis=1)(tmp)

        tmp = Conv3D(8, (3,3,3), kernel_initializer=initializer, padding='same', activation='relu',
                            kernel_regularizer=regularizer, name='upper_conv_7')(tmp)
        tmp = BatchNormalization(axis=1)(tmp)
        tmp = Conv3D(8, (3,3,3), kernel_initializer=initializer, padding='same', activation='relu',
                            kernel_regularizer=regularizer, name='upper_conv_8')(tmp)
        out_up = BatchNormalization(axis=1)(tmp)

        # -------- Lower pathway  ------
        x_down = AveragePooling3D((2, 2, 2))(x)
        tmp = Conv3D(16, (3,3,3), kernel_initializer=initializer, padding='same', activation='relu',
                            kernel_regularizer=regularizer, name='lower_conv_1')(x_down)
        tmp = BatchNormalization(axis=1)(tmp)
        tmp = Conv3D(16, (3,3,3), kernel_initializer=initializer, padding='same', activation='relu',
                            kernel_regularizer=regularizer, name='lower_conv_2')(tmp)
        tmp = BatchNormalization(axis=1)(tmp)
        tmp = MaxPooling3D((2, 2, 2))(tmp)

        tmp = Conv3D(32, (3,3,3), kernel_initializer=initializer, padding='same', activation='relu',
                            kernel_regularizer=regularizer, name='lower_conv_3')(tmp)
        tmp = BatchNormalization(axis=1)(tmp)
        tmp = Conv3D(32, (3,3,3), kernel_initializer=initializer, padding='same', activation='relu',
                            kernel_regularizer=regularizer, name='lower_conv_4')(tmp)
        tmp = BatchNormalization(axis=1)(tmp)
        tmp = MaxPooling3D((2, 2, 2))(tmp)

        tmp = Conv3D(32, (3,3,3), kernel_initializer=initializer, padding='same', activation='relu',
                            kernel_regularizer=regularizer, name='lower_conv_5')(tmp)
        tmp = BatchNormalization(axis=1)(tmp)
        tmp = Conv3D(32, (3,3,3), kernel_initializer=initializer, padding='same', activation='relu',
                            kernel_regularizer=regularizer, name='lower_conv_6')(tmp)
        tmp = BatchNormalization(axis=1)(tmp)
        tmp = MaxPooling3D((2, 2, 2))(tmp)

        tmp = Conv3D(64, (3,3,3), kernel_initializer=initializer, padding='same', activation='relu',
                            kernel_regularizer=regularizer, name='lower_conv_7')(tmp)
        tmp = BatchNormalization(axis=1)(tmp)
        tmp = Conv3D(64, (3,3,3), kernel_initializer=initializer, padding='same', activation='relu',
                            kernel_regularizer=regularizer, name='lower_conv_8')(tmp)
        tmp = BatchNormalization(axis=1)(tmp)

        tmp = UpSampling3D((2, 2, 2), name='upsampling1')(tmp)
        tmp = Conv3D(32, (3,3,3), kernel_initializer=initializer, padding='same', activation='relu',
                            kernel_regularizer=regularizer, name='lower_conv_9')(tmp)
        tmp = BatchNormalization(axis=1)(tmp)
        #
        tmp = UpSampling3D((2, 2, 2), name='upsampling2')(tmp)
        tmp = Conv3D(32, (3,3,3), kernel_initializer=initializer, padding='same', activation='relu',
                            kernel_regularizer=regularizer, name='lower_conv_10')(tmp)
        tmp = BatchNormalization(axis=1)(tmp)

        tmp = UpSampling3D((2, 2, 2), name='upsampling3')(tmp)
        tmp = Conv3D(16, (3,3,3), kernel_initializer=initializer, padding='same', activation='relu',
                            kernel_regularizer=regularizer, name='lower_conv_11')(tmp)
        tmp = BatchNormalization(axis=1)(tmp)

        tmp = UpSampling3D((2, 2, 2), name='upsampling4')(tmp)
        tmp = Conv3D(8, (3,3,3), kernel_initializer=initializer, padding='same', activation='relu',
                            kernel_regularizer=regularizer, name='lower_conv_12')(tmp)
        out_down = BatchNormalization(axis=1)(tmp)

        # Merge two pathways
        concat_layer = Concatenate(axis=1)([out_up, out_down])
        tmp = Conv3D(16, (1, 1, 1), kernel_initializer=initializer, activation='relu', kernel_regularizer=regularizer,
                            name='fully_conv_1')(concat_layer)
        tmp = Conv3D(16, (1, 1, 1), kernel_initializer=initializer, activation='relu', kernel_regularizer=regularizer,
                            name='fully_conv_2')(tmp)
        tmp = Conv3D(num_classes, (1, 1, 1), kernel_initializer=initializer)(tmp)
        y = softmax_activation(tmp)

        # Create and compile model
        model = Model(inputs=x, outputs=y)

        return model, output_shape

    @staticmethod
    def v_net(num_modalities, segment_dimensions, num_classes, BN_last=False, shortcut_input = False,
              l1=0.0,l2=0.0, mask_bool=False):
        """
        U-Net based architecture for segmentation (http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)
            - Convolutional layers: 3x3x3 filters, stride 1, same border
            - Max pooling: 2x2x2
            - Upsampling layers: 2x2x2
            - Activations: ReLU
            - Classification layer: 1x1x1 kernel, and there are as many kernels as classes to be predicted. Element-wise
              softmax as activation.
            - Loss: 3D categorical cross-entropy
            - Optimizer: Adam with lr=0.001, beta_1=0.9, beta_2=0.999 and epsilon=10 ** (-4)
            - Regularization: L1=0.000001, L2=0.0001
            - Weights initialization: He et al normal initialization (https://arxiv.org/abs/1502.01852)

        Returns
        -------
        keras.model
            Compiled Sequential model
        tuple (dim1, dim2, dim3)
            Output shape computed from segment_dimensions and the convolutional architecture
        """
        if not isinstance(segment_dimensions, tuple) or len(segment_dimensions) != 3:
            raise ValueError('segment_dimensions must be a tuple with length 3, specifying the shape of the '
                             'input segment')

        # for dim in segment_dimensions:
        #     assert dim % 32 == 0  # As there are 5 (2, 2, 2) max-poolings, 2 ** 5 is the minimum input size

        # Hyperaparametre values

        initializer = 'he_normal'
        pool_size = (2, 2, 2)

        # Compute input shape, receptive field and output shape after softmax activation
        input_shape =  segment_dimensions + (num_modalities,)
        output_shape = segment_dimensions


        # Activations, regularizers and optimizers
        softmax_activation = Lambda(elementwise_softmax_3d, name='Softmax')
        regularizer = l1_l2(l1=l1, l2=l2)


        # Architecture definition
        # INPUT
        x = Input(shape=input_shape, name='V-net_input')

        # First block (down)
        first_conv = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_initial', padding='same')(x)

        tmp = Activation('relu')(first_conv)
        z1 = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.1', padding='same')(tmp)

        c11 = Conv3D(8, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_conn_1.1', padding='same')(x)
        end_11 = Add()([z1, c11])


        # Second block (down)
        tmp = Activation('relu')(end_11)
        tmp = Conv3D(16, (2, 2, 2), strides=(2, 2, 2), kernel_initializer=initializer, kernel_regularizer=regularizer, name='downpool_1',padding='same')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_2.1', padding='same')(tmp)
        tmp = Activation('relu')(tmp)
        z2 = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_2.2', padding='same')(tmp)

        c21 = MaxPooling3D(pool_size=pool_size, name='pool_1')(end_11)
        c21 = Conv3D(16, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_conn_2.1', padding='same')(c21)

        end_21 = Add()([z2, c21])

        # Third block (down)
        tmp = Activation('relu')(end_21)
        tmp = Conv3D(32, (2, 2, 2), strides=(2,2,2), kernel_initializer=initializer, kernel_regularizer=regularizer, name='downpool_2', padding='same')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_3.1', padding='same')(tmp)
        tmp = Activation('relu')(tmp)
        z3 = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_3.2', padding='same')(tmp)

        c31 = MaxPooling3D(pool_size=pool_size, name='pool_2')(end_21)
        c31 = Conv3D(32, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_conn_3.1', padding='same')(c31)

        end_31 = Add()([z3, c31])


        # Fourth block (down)
        tmp = Activation('relu')(end_31)
        tmp = Conv3D(64, (2, 2, 2), strides=(2,2,2), kernel_initializer=initializer, kernel_regularizer=regularizer, name='downpool_3', padding='same')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_4.1', padding='same')(tmp)
        tmp = Activation('relu')(tmp)
        z4 = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_4.2', padding='same')(tmp)

        c41 = MaxPooling3D(pool_size=pool_size, name='pool_3')(end_31)
        c41 = Conv3D(64, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_conn_4.1', padding='same')(c41)

        end_41 = Add()([z4, c41])

        # Fifth block
        tmp = Activation('relu')(end_41)
        tmp = Conv3D(128, (2, 2, 2), strides=(2,2,2), kernel_initializer=initializer, kernel_regularizer=regularizer, name='downpool_4', padding='same')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(128, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_5.1', padding='same')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(128, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_5.2', padding='same')(tmp)  # inflection point

        c5 = MaxPooling3D(pool_size=pool_size, name='pool_4')(end_41)
        c5 = Conv3D(128, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_conn_5', padding='same')(c5)

        end_5 = Add()([tmp, c5])

        # Fourth block (up)
        tmp = Activation('relu')(end_5)
        tmp = UpSampling3D(size=pool_size, name='up_4')(tmp)
        tmp = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_up_4', padding='same')(tmp)
        tmp = Concatenate(axis=4)([tmp, end_41])
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_4.3', padding='same')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_4.4', padding='same')(tmp)

        c42 = UpSampling3D(size=pool_size, name='up_4conn')(end_5)
        c42 = Conv3D(64, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_conn_4.2', padding='same')(c42)

        end_42 = Add()([tmp, c42])


        # Third block (up)
        tmp = Activation('relu')(end_42)
        tmp = UpSampling3D(size=pool_size, name='up_3')(tmp)
        tmp = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_up_3', padding='same')(tmp)
        tmp = Concatenate(axis=4)([tmp, end_31])
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_3.3', padding='same')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_3.4', padding='same')(tmp)

        c32 = UpSampling3D(size=pool_size, name='up_3conn')(end_42)
        c32 = Conv3D(32, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_conn_3.2', padding='same')(c32)

        end_32 = Add()([tmp, c32])


        # Second block (up)
        tmp = Activation('relu')(end_32)
        tmp = UpSampling3D(size=pool_size, name='up_2')(tmp)
        tmp = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_up_2', padding='same')(tmp)
        tmp = Concatenate(axis=4)([tmp, end_21])
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_2.3', padding='same')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_2.4', padding='same')(tmp)

        c22 = UpSampling3D(size=pool_size, name='up_2conn')(end_32)
        c22 = Conv3D(16, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_conn_2.2', padding='same')(c22)

        end_22 = Add()([tmp, c22])

        # First block (up)
        tmp = Activation('relu')(end_22)
        tmp = UpSampling3D(size=pool_size, name='up_1')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_up_1', padding='same')(tmp)
        tmp = Concatenate(axis=4)([tmp, end_11])
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.3', padding='same')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.4', padding='same')(tmp)

        c12 = UpSampling3D(size=pool_size, name='up_1_conn')(end_22)
        c12 = Conv3D(8, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_conn_1.2', padding='same')(c12)

        end_12 = Add()([tmp,c12])

        # Final convolution
        tmp = Activation('relu')(end_12)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_pre_softmax', padding='same')(tmp)
        in_softmax = Activation('relu')(tmp)

        # Classification layer
        if shortcut_input:
            in_softmax = Concatenate(axis=4)([in_softmax,x])

        classification = Conv3D(num_classes, (1, 1, 1) , kernel_initializer=initializer,
                                name='final_convolution_1x1x1')(in_softmax)
        if mask_bool:
            mask1 = Input(shape=segment_dimensions + (num_classes,), name='V-net_mask1')
            mask2 = Input(shape=segment_dimensions + (num_classes,), name='V-net_mask2')
            y = softmax_activation(classification)
            y = Multiply(name='mask_background')([y,mask1])
            y = Add(name='label_background')([y, mask2])

            model = Model(inputs=[x,mask1,mask2], outputs=y)
        else:
            y = softmax_activation(classification)
            model = Model(inputs=x, outputs=y)

        return model, output_shape


    @staticmethod
    def v_net_BN(num_modalities, segment_dimensions, num_classes, BN_last=False, shortcut_input = False,
              l1=0.0,l2=0.0, mask_bool=False):
        """
        U-Net based architecture for segmentation (http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)
            - Convolutional layers: 3x3x3 filters, stride 1, same border
            - Max pooling: 2x2x2
            - Upsampling layers: 2x2x2
            - Activations: ReLU
            - Classification layer: 1x1x1 kernel, and there are as many kernels as classes to be predicted. Element-wise
              softmax as activation.
            - Loss: 3D categorical cross-entropy
            - Optimizer: Adam with lr=0.001, beta_1=0.9, beta_2=0.999 and epsilon=10 ** (-4)
            - Regularization: L1=0.000001, L2=0.0001
            - Weights initialization: He et al normal initialization (https://arxiv.org/abs/1502.01852)

        Returns
        -------
        keras.model
            Compiled Sequential model
        tuple (dim1, dim2, dim3)
            Output shape computed from segment_dimensions and the convolutional architecture
        """
        if not isinstance(segment_dimensions, tuple) or len(segment_dimensions) != 3:
            raise ValueError('segment_dimensions must be a tuple with length 3, specifying the shape of the '
                             'input segment')

        # for dim in segment_dimensions:
        #     assert dim % 32 == 0  # As there are 5 (2, 2, 2) max-poolings, 2 ** 5 is the minimum input size

        # Hyperaparametre values

        initializer = 'he_normal'
        pool_size = (2, 2, 2)

        # Compute input shape, receptive field and output shape after softmax activation
        input_shape =  segment_dimensions + (num_modalities,)
        output_shape = segment_dimensions


        # Activations, regularizers and optimizers
        softmax_activation = Lambda(elementwise_softmax_3d, name='Softmax')
        regularizer = l1_l2(l1=l1, l2=l2)


        # Architecture definition
        # INPUT
        x = Input(shape=input_shape, name='V-net_input')
        first_conv = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_initial', padding='same')(x)

        # First block (down)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.1')(first_conv)
        tmp = Activation('relu')(tmp)
        z1 = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.1', padding='same')(tmp)

        c11 = Conv3D(8, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_conn_1.1', padding='same')(first_conv)
        end_11 = Add()([z1, c11])


        # Second block (down)
        tmp = BatchNormalization(axis=4, name='batch_norm_2.1')(end_11)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(16, (2, 2, 2), strides=(2, 2, 2), kernel_initializer=initializer, kernel_regularizer=regularizer, name='downpool_1',padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_2.2')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_2.1', padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_2.3')(tmp)
        tmp = Activation('relu')(tmp)
        z2 = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_2.2', padding='same')(tmp)

        c21 = MaxPooling3D(pool_size=pool_size, name='pool_1')(end_11)
        c21 = Conv3D(16, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_conn_2.1', padding='same')(c21)

        end_21 = Add()([z2, c21])

        # Third block (down)
        tmp = BatchNormalization(axis=4, name='batch_norm_3.1')(end_21)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(32, (2, 2, 2), strides=(2,2,2), kernel_initializer=initializer, kernel_regularizer=regularizer, name='downpool_2', padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_3.2')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_3.1', padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_3.3')(tmp)
        tmp = Activation('relu')(tmp)
        z3 = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_3.2', padding='same')(tmp)

        c31 = MaxPooling3D(pool_size=pool_size, name='pool_2')(end_21)
        c31 = Conv3D(32, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_conn_3.1', padding='same')(c31)

        end_31 = Add()([z3, c31])


        # Fourth block (down)
        tmp = BatchNormalization(axis=4, name='batch_norm_4.1')(end_31)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(64, (2, 2, 2), strides=(2,2,2), kernel_initializer=initializer, kernel_regularizer=regularizer, name='downpool_3', padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_4.2')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_4.1', padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_4.3')(tmp)
        tmp = Activation('relu')(tmp)
        z4 = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_4.2', padding='same')(tmp)

        c41 = MaxPooling3D(pool_size=pool_size, name='pool_3')(end_31)
        c41 = Conv3D(64, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_conn_4.1', padding='same')(c41)

        end_41 = Add()([z4, c41])

        # Fifth block
        tmp = BatchNormalization(axis=4, name='batch_norm_5.1')(end_41)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(128, (2, 2, 2), strides=(2,2,2), kernel_initializer=initializer, kernel_regularizer=regularizer, name='downpool_4', padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_5.2')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(128, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_5.1', padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_5.3')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(128, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_5.2', padding='same')(tmp)  # inflection point

        c5 = MaxPooling3D(pool_size=pool_size, name='pool_4')(end_41)
        c5 = Conv3D(128, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_conn_5', padding='same')(c5)

        end_5 = Add()([tmp, c5])

        # Fourth block (up)
        tmp = BatchNormalization(axis=4, name='batch_norm_4.4')(end_5)
        tmp = Activation('relu')(tmp)
        tmp = UpSampling3D(size=pool_size, name='up_4')(tmp)
        tmp = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_up_4', padding='same')(tmp)
        tmp = Concatenate(axis=4)([tmp, end_41])
        tmp = BatchNormalization(axis=4, name='batch_norm_4.5')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_4.3', padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_4.6')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_4.4', padding='same')(tmp)

        c42 = UpSampling3D(size=pool_size, name='up_4conn')(end_5)
        c42 = Conv3D(64, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_conn_4.2', padding='same')(c42)

        end_42 = Add()([tmp, c42])


        # Third block (up)
        tmp = BatchNormalization(axis=4, name='batch_norm_3.4')(end_42)
        tmp = Activation('relu')(tmp)
        tmp = UpSampling3D(size=pool_size, name='up_3')(tmp)
        tmp = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_up_3', padding='same')(tmp)
        tmp = Concatenate(axis=4)([tmp, end_31])
        tmp = BatchNormalization(axis=4, name='batch_norm_3.5')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_3.3', padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_3.6')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_3.4', padding='same')(tmp)

        c32 = UpSampling3D(size=pool_size, name='up_3conn')(end_42)
        c32 = Conv3D(32, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_conn_3.2', padding='same')(c32)

        end_32 = Add()([tmp, c32])


        # Second block (up)
        tmp = BatchNormalization(axis=4, name='batch_norm_2.4')(end_32)
        tmp = Activation('relu')(tmp)
        tmp = UpSampling3D(size=pool_size, name='up_2')(tmp)
        tmp = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_up_2', padding='same')(tmp)
        tmp = Concatenate(axis=4)([tmp, end_21])
        tmp = BatchNormalization(axis=4, name='batch_norm_2.5')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_2.3', padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_2.6')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_2.4', padding='same')(tmp)

        c22 = UpSampling3D(size=pool_size, name='up_2conn')(end_32)
        c22 = Conv3D(16, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_conn_2.2', padding='same')(c22)

        end_22 = Add()([tmp, c22])

        # First block (up)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.4')(end_22)
        tmp = Activation('relu')(tmp)
        tmp = UpSampling3D(size=pool_size, name='up_1')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_up_1', padding='same')(tmp)
        tmp = Concatenate(axis=4)([tmp, end_11])
        tmp = BatchNormalization(axis=4, name='batch_norm_1.5')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.3', padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.6')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.4', padding='same')(tmp)

        c12 = UpSampling3D(size=pool_size, name='up_1_conn')(end_22)
        c12 = Conv3D(8, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_conn_1.2', padding='same')(c12)

        end_12 = Add()([tmp,c12])

        # Final convolution
        tmp = BatchNormalization(axis=4, name='batch_norm_1.7')(end_12)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_pre_softmax', padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_pre_softmax')(tmp)
        in_softmax = Activation('relu')(tmp)

        # Classification layer
        if shortcut_input:
            in_softmax = Concatenate(axis=4)([in_softmax,x])

        classification = Conv3D(num_classes, (1, 1, 1) , kernel_initializer=initializer,
                                name='final_convolution_1x1x1')(in_softmax)
        if BN_last:
            classification = BatchNormalization(axis=4, name='final_batch_norm')(classification)

        if mask_bool:
            mask1 = Input(shape=segment_dimensions + (num_classes,), name='V-net_mask1')
            mask2 = Input(shape=segment_dimensions + (num_classes,), name='V-net_mask2')
            y = softmax_activation(classification)
            y = Multiply(name='mask_background')([y,mask1])
            y = Add(name='label_background')([y, mask2])

            model = Model(inputs=[x,mask1,mask2], outputs=y)
        else:
            y = softmax_activation(classification)
            model = Model(inputs=x, outputs=y)

        return model, output_shape

    @staticmethod
    def Masked_v_net_BN(num_modalities, segment_dimensions, num_classes, shortcut_input=False, l1=0.0, l2=0.0):
        """
        U-Net based architecture for segmentation (http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)
            - Convolutional layers: 3x3x3 filters, stride 1, same border
            - Max pooling: 2x2x2
            - Upsampling layers: 2x2x2
            - Activations: ReLU
            - Classification layer: 1x1x1 kernel, and there are as many kernels as classes to be predicted. Element-wise
              softmax as activation.
            - Loss: 3D categorical cross-entropy
            - Optimizer: Adam with lr=0.001, beta_1=0.9, beta_2=0.999 and epsilon=10 ** (-4)
            - Regularization: L1=0.000001, L2=0.0001
            - Weights initialization: He et al normal initialization (https://arxiv.org/abs/1502.01852)

        Returns
        -------
        keras.model
            Compiled Sequential model
        tuple (dim1, dim2, dim3)
            Output shape computed from segment_dimensions and the convolutional architecture
        """
        if not isinstance(segment_dimensions, tuple) or len(segment_dimensions) != 3:
            raise ValueError('segment_dimensions must be a tuple with length 3, specifying the shape of the '
                             'input segment')

        # for dim in segment_dimensions:
        #     assert dim % 32 == 0  # As there are 5 (2, 2, 2) max-poolings, 2 ** 5 is the minimum input size

        # Hyperaparametre values

        initializer = 'he_normal'
        pool_size = (2, 2, 2)

        # Compute input shape, receptive field and output shape after softmax activation
        input_shape = segment_dimensions + (num_modalities,)
        output_shape = segment_dimensions

        # Activations, regularizers and optimizers
        softmax_activation = Lambda(elementwise_softmax_3d, name='Softmax')
        regularizer = l1_l2(l1=l1, l2=l2)

        # Architecture definition
        # INPUT
        x = Input(shape=input_shape, name='V-net_input')
        first_conv = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer,
                            name='conv_initial', padding='same')(x)

        # First block (down)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.1')(first_conv)
        tmp = Activation('relu')(tmp)
        z1 = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.1',
                    padding='same')(tmp)

        c11 = Conv3D(8, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_conn_1.1',
                     padding='same')(first_conv)
        end_11 = Add()([z1, c11])

        # Second block (down)
        tmp = BatchNormalization(axis=4, name='batch_norm_2.1')(end_11)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(16, (2, 2, 2), strides=(2, 2, 2), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='downpool_1', padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_2.2')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_2.1',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_2.3')(tmp)
        tmp = Activation('relu')(tmp)
        z2 = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_2.2',
                    padding='same')(tmp)

        c21 = MaxPooling3D(pool_size=pool_size, name='pool_1')(end_11)
        c21 = Conv3D(16, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_conn_2.1', padding='same')(c21)

        end_21 = Add()([z2, c21])

        # Third block (down)
        tmp = BatchNormalization(axis=4, name='batch_norm_3.1')(end_21)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(32, (2, 2, 2), strides=(2, 2, 2), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='downpool_2', padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_3.2')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_3.1',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_3.3')(tmp)
        tmp = Activation('relu')(tmp)
        z3 = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_3.2',
                    padding='same')(tmp)

        c31 = MaxPooling3D(pool_size=pool_size, name='pool_2')(end_21)
        c31 = Conv3D(32, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_conn_3.1', padding='same')(c31)

        end_31 = Add()([z3, c31])

        # Fourth block (down)
        tmp = BatchNormalization(axis=4, name='batch_norm_4.1')(end_31)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(64, (2, 2, 2), strides=(2, 2, 2), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='downpool_3', padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_4.2')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_4.1',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_4.3')(tmp)
        tmp = Activation('relu')(tmp)
        z4 = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_4.2',
                    padding='same')(tmp)

        c41 = MaxPooling3D(pool_size=pool_size, name='pool_3')(end_31)
        c41 = Conv3D(64, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_conn_4.1', padding='same')(c41)

        end_41 = Add()([z4, c41])

        # Fifth block
        tmp = BatchNormalization(axis=4, name='batch_norm_5.1')(end_41)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(128, (2, 2, 2), strides=(2, 2, 2), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='downpool_4', padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_5.2')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(128, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_5.1',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_5.3')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(128, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_5.2',
                     padding='same')(tmp)  # inflection point

        c5 = MaxPooling3D(pool_size=pool_size, name='pool_4')(end_41)
        c5 = Conv3D(128, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_conn_5',
                    padding='same')(c5)

        end_5 = Add()([tmp, c5])

        # Fourth block (up)
        tmp = BatchNormalization(axis=4, name='batch_norm_4.4')(end_5)
        tmp = Activation('relu')(tmp)
        tmp = UpSampling3D(size=pool_size, name='up_4')(tmp)
        tmp = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_up_4',
                     padding='same')(tmp)
        tmp = Concatenate(axis=4)([tmp, end_41])
        tmp = BatchNormalization(axis=4, name='batch_norm_4.5')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_4.3',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_4.6')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_4.4',
                     padding='same')(tmp)

        c42 = UpSampling3D(size=pool_size, name='up_4conn')(end_5)
        c42 = Conv3D(64, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_conn_4.2', padding='same')(c42)

        end_42 = Add()([tmp, c42])

        # Third block (up)
        tmp = BatchNormalization(axis=4, name='batch_norm_3.4')(end_42)
        tmp = Activation('relu')(tmp)
        tmp = UpSampling3D(size=pool_size, name='up_3')(tmp)
        tmp = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_up_3',
                     padding='same')(tmp)
        tmp = Concatenate(axis=4)([tmp, end_31])
        tmp = BatchNormalization(axis=4, name='batch_norm_3.5')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_3.3',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_3.6')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_3.4',
                     padding='same')(tmp)

        c32 = UpSampling3D(size=pool_size, name='up_3conn')(end_42)
        c32 = Conv3D(32, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_conn_3.2', padding='same')(c32)

        end_32 = Add()([tmp, c32])

        # Second block (up)
        tmp = BatchNormalization(axis=4, name='batch_norm_2.4')(end_32)
        tmp = Activation('relu')(tmp)
        tmp = UpSampling3D(size=pool_size, name='up_2')(tmp)
        tmp = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_up_2',
                     padding='same')(tmp)
        tmp = Concatenate(axis=4)([tmp, end_21])
        tmp = BatchNormalization(axis=4, name='batch_norm_2.5')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_2.3',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_2.6')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_2.4',
                     padding='same')(tmp)

        c22 = UpSampling3D(size=pool_size, name='up_2conn')(end_32)
        c22 = Conv3D(16, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_conn_2.2', padding='same')(c22)

        end_22 = Add()([tmp, c22])

        # First block (up)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.4')(end_22)
        tmp = Activation('relu')(tmp)
        tmp = UpSampling3D(size=pool_size, name='up_1')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_up_1',
                     padding='same')(tmp)
        tmp = Concatenate(axis=4)([tmp, end_11])
        tmp = BatchNormalization(axis=4, name='batch_norm_1.5')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.3',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.6')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.4',
                     padding='same')(tmp)

        c12 = UpSampling3D(size=pool_size, name='up_1_conn')(end_22)
        c12 = Conv3D(8, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_conn_1.2',
                     padding='same')(c12)

        end_12 = Add()([tmp, c12])

        # Final convolution
        tmp = BatchNormalization(axis=4, name='batch_norm_1.7')(end_12)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_pre_softmax', padding='same')(tmp)

        if shortcut_input:
            tmp = Concatenate(axis=4)([tmp, x])

        tmp = BatchNormalization(axis=4, name='batch_norm_pre_softmax')(tmp)
        in_softmax = Activation('relu')(tmp)

        classification = Conv3D(num_classes, (1, 1, 1), kernel_initializer=initializer,
                                name='final_convolution_1x1x1')(in_softmax)

        y = softmax_activation(classification)

        mask = Input(shape=segment_dimensions + (1,), name='V-net_mask1')
        mask_rep = Lambda(repeat_channels(num_classes), output_shape=repeat_channels_shape(num_classes),
                       name='repeat_mask')(mask)
        cmp_mask = Lambda(complementary_mask, output_shape=segment_dimensions + (1,),
                          name='complementary_tumor_mask')(mask)

        cmp_mask = Concatenate()([cmp_mask] + [Lambda(lambda x: K.zeros_like(x))(cmp_mask) for _ in range(num_classes-1)])

        y = Multiply(name='mask_background')([y, mask_rep])
        y = Add(name='label_background')([y, cmp_mask])

        model = Model(inputs=[x, mask], outputs=y)

        return model, output_shape


class SCAE(object):
    """ Interface that allows you to save and load models and their weights """

    # ------------------------------------------------ CONSTANTS ------------------------------------------------ #

    DEFAULT_MODEL = 'v_net'

    # ------------------------------------------------- METHODS ------------------------------------------------- #

    @classmethod
    def get_model(cls, num_modalities, segment_dimensions, CAE, model_name=None, **kwargs):
        """
        Returns the compiled model specified by its name.
        If no name is given, the default model is returned, which corresponds to the
        hand-picked model that performs the best.

        Parameters
        ----------
        num_modalities : int
            Number of modalities used as input channels
        segment_dimensions : tuple
            Tuple with 3 elements that specify the shape of the input segments
        num_classes : int
            Number of output classes to be predicted in the segmentation
        model_name : [Optional] String
            Name of the model to be returned
        weights_filename: [Optional] String
            Path to the H5 file containing the weights of the trained model. The user must ensure the
            weights correspond to the model to be loaded.

        Returns
        -------
        keras.Model
            Compiled keras model
        tuple
            Tuple with output size (necessary to adjust the ground truth matrix's size),
            or None if output size is the sames as segment_dimensions.

        Raises
        ------
        TypeError
            If model_name is not a String or None
        ValueError
            If the model specified by model_name does not exist
        """
        if model_name is None:
            return cls.get_model(num_modalities, segment_dimensions, CAE, model_name=cls.DEFAULT_MODEL, **kwargs)

        if isinstance(model_name, str):
            try:

                model_getter = cls.__dict__[model_name]
                return model_getter.__func__(num_modalities, segment_dimensions, CAE, **kwargs)
            except KeyError:
                raise ValueError('Model {} does not exist. Use the class method {} to know the available model names'
                                 .format(model_name, 'get_model_names'))
        else:
            raise TypeError('model_name must be a String/Unicode sequence or None')

    @classmethod
    def get_model_names(cls):
        """
        List of available models' names

        Returns
        -------
        List
            Names of the available model names to be used in get_model(model_name) method
        """

        def filter_functions(attr_name):
            tmp = ('model' not in attr_name) and ('MODEL' not in attr_name)
            return tmp and ('__' not in attr_name) and ('BASE' not in attr_name)

        return filter(filter_functions, cls.__dict__.keys())

    @staticmethod
    def compile_scae(model, lr=None):
        '''
        Compile the model
        '''

        # Optimizer values
        lr = 0.02 if lr is None else lr
        beta_1 = 0.9
        beta_2 = 0.999
        epsilon = 10 ** (-8)
        optimizer = Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, clipnorm=1.)

        model.compile(
            optimizer=optimizer,
            loss=[lambda y_true, y_pred: y_pred],
        )

        return model

    @staticmethod
    def stacked_CAE_masked(num_modalities, segment_dimensions, CAE, pre_train_CAE=True, stddev=0):
        """
        Stacked convolutional autoencoder

        Returns
        -------
        keras.model
            Compiled Sequential model
        tuple (dim1, dim2, dim3)
            Output shape computed from segment_dimensions and the convolutional architecture
        """

        if not isinstance(segment_dimensions, tuple) or len(segment_dimensions) != 3:
            raise ValueError('segment_dimensions must be a tuple with length 3, specifying the shape of the '
                             'input segment')
        if not isinstance(CAE, list):
            if isinstance(CAE, dict):
                CAE = [CAE]
            else:
                raise ValueError("Please, specify a valid dictionary for the definiton of CAE")

        # Compute input shape, receptive field and output shape
        input_shape = segment_dimensions + (num_modalities,)
        input_layer = Input(shape=input_shape, name='SCAE_input')

        pool_size = (2, 2, 2)
        n_channels = num_modalities

        n_CAE = len(CAE)
        if pre_train_CAE:
            mask_shape = segment_dimensions + (1,)
            mask_layer = Input(shape=mask_shape, name='SCAE_mask')

            input_cae_model = input_layer
            output_mask_pool = mask_layer
            for it_cae_dict, cae_dict in enumerate(CAE[:n_CAE-1]):
                cae_model = SCAE.CAE(n_channels, K.int_shape(input_cae_model)[1:-1], cae_dict, name_suffix=str(it_cae_dict), stddev=stddev, pre_train_CAE=False)

                n_channels = cae_dict['n_filters']
                input_cae_model = cae_model(input_cae_model)
                input_cae_model = MaxPooling3D(pool_size=pool_size, name='pool_' + str(it_cae_dict))(input_cae_model)
                output_mask_pool = MaxPooling3D(pool_size=pool_size, name='pool_mask_' + str(it_cae_dict))(output_mask_pool)
                output_mask_rep = Lambda(repeat_channels(n_channels), output_shape=repeat_channels_shape(n_channels),
                                           name='repeat_channel_mask_' + str(it_cae_dict))(output_mask_pool)
                input_cae_model = Multiply(name='multiply_in' + str(it_cae_dict))([input_cae_model, output_mask_rep])

            cae_model = SCAE.CAE(n_channels, K.int_shape(input_cae_model)[1:-1], CAE[-1], name_suffix=str(n_CAE-1), stddev=stddev, pre_train_CAE=True)

            output_cae_model = cae_model(input_cae_model)
            output_mask_rep = Lambda(repeat_channels(n_channels), output_shape=repeat_channels_shape(n_channels),
                                       name = 'repeat_channel_mask_' + str(n_CAE-1))(output_mask_pool)
            output_cae_model = Multiply(name = 'multiply_out' + str(n_CAE-1))([output_cae_model, output_mask_rep])


            loss_out = Lambda(scae_mean_squared_error_masked,
                              output_shape=(1,),
                              name='joint_loss')([input_cae_model,output_cae_model,output_mask_pool])

            model = Model(inputs=[input_layer,mask_layer], outputs=loss_out)

        else:
            output_cae_model = input_layer
            for it_cae_dict, cae_dict in enumerate(CAE):
                cae_model = SCAE.CAE(n_channels, K.int_shape(output_cae_model)[1:-1], cae_dict,
                                                  name_suffix=str(it_cae_dict), stddev=stddev, pre_train_CAE=False)

                n_channels = cae_dict['n_filters']
                output_cae_model = cae_model(output_cae_model)
                output_cae_model = MaxPooling3D(pool_size=pool_size, name='pool_' + str(it_cae_dict))(output_cae_model)


            model = Model(inputs=input_layer, outputs=output_cae_model)

        return model


    @staticmethod
    def CAE(num_modalities, segment_dimensions, CAE, name_suffix = '', stddev = 0, pre_train_CAE = False, **kwargs):
        """
        Convolutional autoencoder
            - 2 Convolutional layers with 5x5x5 kernels, stride 1 and no padding.
            - Activations: ReLU
            - Loss: 3D categorical cross-entropy
            - Optimizer: RMSProp with the following parameters: learning_rate=0.001, rho=0.9, epsilon=10**(-4)
            - Regularization: L1=0.000001, L2=0.0001
            - Weights initialization: He et al normal initialization (https://arxiv.org/abs/1502.01852)
        Returns
        -------
        keras.model
            Compiled Sequential model
        tuple (dim1, dim2, dim3)
            Output shape computed from segment_dimensions and the convolutional architecture
        """


        if not isinstance(segment_dimensions, tuple) or len(segment_dimensions) != 3:
            raise ValueError('segment_dimensions must be a tuple with length 3, specifying the shape of the '
                             'input segment')

        #Check CAE keys
        assert 'n_filters' in CAE.keys()
        assert 'train_flag' in CAE.keys()
        assert 'weights_filename' in CAE.keys()

        # Compute input shape, receptive field and output shape after softmax activation
        input_shape =  segment_dimensions + (num_modalities,)

        # Hyperaparametre values
        L1_reg = 0.000001
        L2_reg = 0.0001
        initializer = 'he_normal'
        regularizer = l1_l2(l1=L1_reg, l2=L2_reg)


        # Architecture
        input_layer = Input(shape=input_shape, name='CAE_'+name_suffix+'_input')
        noisy_input = GaussianNoise(stddev=stddev)(input_layer)
        env_conv = Conv3D(CAE['n_filters'], (3, 3, 3), kernel_initializer=initializer, padding='same',
                                 kernel_regularizer=regularizer, name='encoder_' + name_suffix,
                                 trainable=CAE['train_flag'])

        hidden_layer = env_conv(noisy_input)
        # hidden_layer = BatchNormalization(axis=4, name = 'BN_' + name_suffix)(hidden_layer)
        hidden_layer = Activation('relu', name = 'ReLU_'+name_suffix)(hidden_layer)

        output_layer = Conv3D(num_modalities, (3,3,3), padding='same', #activation = 'sigmoid',
                              kernel_regularizer=regularizer, name='decoder_' + name_suffix,
                              trainable=CAE['train_flag'])(hidden_layer)

        # output_layer = Conv3D_tied(num_modalities, tied_to=env_conv,  padding='same',
        #                            kernel_regularizer=regularizer, name='decoder_'+name_suffix,
        #                            )(hidden_layer)

        if pre_train_CAE:
            model = Model(inputs=input_layer, outputs=output_layer, name='CAE_'+name_suffix)
            if CAE['weights_filename'] is not None:
                try:
                    weight_value_tuples = []
                    symbolic_weights = [w for w in model.weights]
                    symbolic_weights_name = [s.name for s in symbolic_weights]

                    f = h5py.File(CAE['weights_filename'], 'r')
                    if 'layer_names' not in f.attrs and 'model_weights' in f:
                        f = f['model_weights']
                    # weight_names = f['CAE_' + name_suffix].attrs['weight_names'].tolist()
                        print('CAE pre-train ' + str(name_suffix))
                    print(symbolic_weights_name)
                    print(f['CAE_' + name_suffix].name)
                    weight_values = [f['CAE_' + name_suffix][weight_name] for weight_name in symbolic_weights_name]

                    # Set values.
                    for i in range(len(weight_values)):
                        weight_value_tuples.append((symbolic_weights[i],
                                                    weight_values[i]))
                    K.batch_set_value(weight_value_tuples)
                except:
                    raise ValueError('Weights could not be loaded for CAE_' + name_suffix)

            else:
                warnings.warn('CAE_' + name_suffix + ' acts as a CAE without preloading weights. Trainable = ' +
                              str(CAE['train_flag']))
        else:
            model = Model(inputs=input_layer, outputs=hidden_layer, name='CAE_'+name_suffix)

            if CAE['weights_filename'] is not None:
                try:
                    weight_value_tuples = []
                    symbolic_weights = [w for w in model.weights]
                    symbolic_weights_name = [s.name for s in symbolic_weights]

                    f = h5py.File(CAE['weights_filename'], 'r')
                    if 'layer_names' not in f.attrs and 'model_weights' in f:
                        f = f['model_weights']
                    # weight_names = f['CAE_' + name_suffix].attrs['weight_names'].tolist()
                    print('CAE trained ' + str(name_suffix))
                    print(symbolic_weights_name)
                    print(f['CAE_' + name_suffix].name)
                    weight_values = [f['CAE_' + name_suffix][weight_name] for weight_name in symbolic_weights_name]
                    # Set values.
                    for i in range(len(weight_values)):
                        weight_value_tuples.append((symbolic_weights[i],
                                                    weight_values[i]))
                    K.batch_set_value(weight_value_tuples)
                except:
                    raise ValueError('Weights could not be loaded for CAE_'+name_suffix)

            else:
                warnings.warn('CAE_' + name_suffix + ' acts as a ConvLayer without preloading weights. Trainable = ' +
                              str(CAE['train_flag']))

        return model


class BraTS_models(object):
    """ Interface that allows you to save and load models and their weights """

    # ------------------------------------------------ CONSTANTS ------------------------------------------------ #

    DEFAULT_MODEL = 'v_net'

    # ------------------------------------------------- METHODS ------------------------------------------------- #

    @classmethod
    def get_model(cls, num_modalities, segment_dimensions, num_classes, model_name=None,
                  **kwargs):
        """
        Returns the compiled model specified by its name.
        If no name is given, the default model is returned, which corresponds to the
        hand-picked model that performs the best.

        Parameters
        ----------
        num_modalities : int
            Number of modalities used as input channels
        segment_dimensions : tuple
            Tuple with 3 elements that specify the shape of the input segments
        num_classes : int
            Number of output classes to be predicted in the segmentation
        model_name : [Optional] String
            Name of the model to be returned
        weights_filename: [Optional] String
            Path to the H5 file containing the weights of the trained model. The user must ensure the
            weights correspond to the model to be loaded.

        Returns
        -------
        keras.Model
            Compiled keras model
        tuple
            Tuple with output size (necessary to adjust the ground truth matrix's size),
            or None if output size is the sames as segment_dimensions.

        Raises
        ------
        TypeError
            If model_name is not a String or None
        ValueError
            If the model specified by model_name does not exist
        """
        if model_name is None:
            return cls.get_model(num_modalities, segment_dimensions, num_classes, model_name=cls.DEFAULT_MODEL,
                                 **kwargs)
        if isinstance(model_name, str):
            try:

                model_getter = cls.__dict__[model_name]
                return model_getter.__func__(num_modalities, segment_dimensions, num_classes,
                                             **kwargs)
            except KeyError:
                raise ValueError('Model {} does not exist. Use the class method {} to know the available model names'
                                 .format(model_name, 'get_model_names'))
        else:
            raise TypeError('model_name must be a String/Unicode sequence or None')

    @classmethod
    def get_model_names(cls):
        """
        List of available models' names

        Returns
        -------
        List
            Names of the available model names to be used in get_model(model_name) method
        """

        def filter_functions(attr_name):
            tmp = ('model' not in attr_name) and ('MODEL' not in attr_name)
            return tmp and ('__' not in attr_name) and ('BASE' not in attr_name)

        return filter(filter_functions, cls.__dict__.keys())


    @staticmethod
    def compile(model, lr=0.0005, optimizer_name='Adam', model_name='full'):
        if optimizer_name == 'Adam':
            beta_1 = 0.9
            beta_2 = 0.999
            epsilon = 10 ** (-8)
            optimizer = Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, clipnorm=1.)

        elif optimizer_name == 'RMSProp':
            rho = 0.9
            epsilon = 10 ** (-8)
            decay = 0.0
            optimizer = RMSprop(lr=lr, rho=rho, epsilon=epsilon, decay=decay)

        elif optimizer_name == 'SGD':
            optimizer = SGD(lr=lr, momentum=0.0, decay=0.0, nesterov=False)

        else:
            raise ValueError('Please, specify a valid optimizer')

        metrics = {'label_mask_background': ['accuracy', dice_1],
                   'label_seg_background': ['accuracy', dice_whole, dice_core, dice_enhance]}

        if model_name == 'full':
            loss = [dice_cost, lambda y_true, y_pred: y_pred, 'mse']
            loss_weights = [0.1, 1, 0.0001]
        elif model_name =='segmentation':
            loss = [lambda y_true, y_pred: y_pred]
            loss_weights = [1]
        elif model_name == 'survival':
            loss='mse'
            metrics = None
            loss_weights = None
        else:
            raise ValueError('Please, specify a valid model name')

        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            loss_weights=loss_weights

        )

        return model

    @staticmethod
    def create_vnet(input, num_classes, net_name, trainable=True, survival=False):
        # Hyperaparametre values

        L1_reg = 0.0001  # initial 0.000001
        L2_reg = 0.01  # initial  0.0001
        initializer = 'he_normal'
        pool_size = (2, 2, 2)
        momentum = 0.7

        # Activations, regularizers and optimizers
        softmax_activation = Lambda(elementwise_softmax_3d, name=net_name + 'Softmax')
        regularizer = l1_l2(l1=L1_reg, l2=L2_reg)

        # First block (down)
        first_conv = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer,
                            name=net_name + 'conv_initial', trainable=trainable,
                            padding='same')(input)
        tmp = BatchNormalization(axis=4, momentum=momentum, name=net_name + 'batch_norm_1.1', trainable=trainable)(first_conv)
        tmp = Activation('relu')(tmp)
        z1 = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer,
                    name=net_name + 'conv_1.1', padding='same', trainable=trainable)(tmp)

        c11 = Conv3D(8, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name=net_name + 'conv_conn_1.1', padding='same', trainable=trainable)(input)
        end_11 = Add()([z1, c11])

        # Second block (down)
        tmp = BatchNormalization(axis=4, momentum=momentum, name=net_name + 'batch_norm_2.1', trainable=trainable)(end_11)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(16, (2, 2, 2), strides=(2, 2, 2), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name=net_name + 'downpool_1', padding='same', trainable=trainable)(
            tmp)
        tmp = BatchNormalization(axis=4, momentum=momentum, name=net_name + 'batch_norm_2.2', trainable=trainable)(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name=net_name + 'conv_2.1', padding='same', trainable=trainable)(tmp)
        tmp = BatchNormalization(axis=4, momentum=momentum, name=net_name + 'batch_norm_2.3', trainable=trainable)(tmp)
        tmp = Activation('relu')(tmp)
        z2 = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer,
                    name=net_name + 'conv_2.2', padding='same', trainable=trainable)(tmp)

        c21 = MaxPooling3D(pool_size=pool_size, name=net_name + 'pool_1')(end_11)
        c21 = Conv3D(16, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name=net_name + 'conv_conn_2.1', padding='same', trainable=trainable)(c21)

        end_21 = Add()([z2, c21])

        # Third block (down)
        tmp = BatchNormalization(axis=4, momentum=momentum, name=net_name + 'batch_norm_3.1', trainable=trainable)(end_21)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(32, (2, 2, 2), strides=(2, 2, 2), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name=net_name + 'downpool_2', padding='same', trainable=trainable)(
            tmp)
        tmp = BatchNormalization(axis=4, momentum=momentum, name=net_name + 'batch_norm_3.2', trainable=trainable)(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name=net_name + 'conv_3.1', padding='same', trainable=trainable)(tmp)
        tmp = BatchNormalization(axis=4, momentum=momentum, name=net_name + 'batch_norm_3.3', trainable=trainable)(tmp)
        tmp = Activation('relu')(tmp)
        z3 = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer,
                    name=net_name + 'conv_3.2', padding='same', trainable=trainable)(tmp)

        c31 = MaxPooling3D(pool_size=pool_size, name=net_name + 'pool_2')(end_21)
        c31 = Conv3D(32, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name=net_name + 'conv_conn_3.1', padding='same', trainable=trainable)(c31)

        end_31 = Add()([z3, c31])

        # Fourth block (down)
        tmp = BatchNormalization(axis=4, momentum=momentum, name=net_name + 'batch_norm_4.1', trainable=trainable)(end_31)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(64, (2, 2, 2), strides=(2, 2, 2), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name=net_name + 'downpool_3', padding='same', trainable=trainable)(
            tmp)
        tmp = BatchNormalization(axis=4, momentum=momentum, name=net_name + 'batch_norm_4.2', trainable=trainable)(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name=net_name + 'conv_4.1', padding='same', trainable=trainable)(tmp)
        tmp = BatchNormalization(axis=4, momentum=momentum, name=net_name + 'batch_norm_4.3', trainable=trainable)(tmp)
        tmp = Activation('relu')(tmp)
        z4 = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer,
                    name=net_name + 'conv_4.2', padding='same', trainable=trainable)(tmp)

        c41 = MaxPooling3D(pool_size=pool_size, name=net_name + 'pool_3')(end_31)
        c41 = Conv3D(64, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name=net_name + 'conv_conn_4.1', padding='same', trainable=trainable)(c41)

        end_41 = Add()([z4, c41])

        # Fifth block
        tmp = BatchNormalization(axis=4, momentum=momentum, name=net_name + 'batch_norm_5.1', trainable=trainable)(end_41)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(128, (2, 2, 2), strides=(2, 2, 2), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name=net_name + 'downpool_4', padding='same', trainable=trainable)(
            tmp)
        tmp = BatchNormalization(axis=4, momentum=momentum, name=net_name + 'batch_norm_5.2', trainable=trainable)(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(128, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name=net_name + 'conv_5.1', padding='same', trainable=trainable)(tmp)
        tmp = BatchNormalization(axis=4, momentum=momentum, name=net_name + 'batch_norm_5.3', trainable=trainable)(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(128, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name=net_name + 'conv_5.2', padding='same', trainable=trainable)(
            tmp)  # inflection point

        c5 = MaxPooling3D(pool_size=pool_size, name=net_name + 'pool_4')(end_41)
        c5 = Conv3D(128, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                    name=net_name + 'conv_conn_5', padding='same', trainable=trainable)(c5)

        end_5 = Add()([tmp, c5])

        # Fourth block (up)
        tmp = BatchNormalization(axis=4, momentum=momentum, name=net_name + 'batch_norm_4.4', trainable=trainable)(end_5)
        tmp = Activation('relu')(tmp)
        tmp = UpSampling3D(size=pool_size, name=net_name + 'up_4')(tmp)
        tmp = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name=net_name + 'conv_up_4', padding='same', trainable=trainable)(tmp)
        tmp = Concatenate(axis=4)([tmp, end_41])
        tmp = BatchNormalization(axis=4, momentum=momentum, name=net_name + 'batch_norm_4.5', trainable=trainable)(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name=net_name + 'conv_4.3', padding='same', trainable=trainable)(tmp)
        tmp = BatchNormalization(axis=4, momentum=momentum, name=net_name + 'batch_norm_4.6', trainable=trainable)(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name=net_name + 'conv_4.4', padding='same', trainable=trainable)(tmp)

        c42 = UpSampling3D(size=pool_size, name=net_name + 'up_4conn')(end_5)
        c42 = Conv3D(64, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name=net_name + 'conv_conn_4.2', padding='same', trainable=trainable)(c42)

        end_42 = Add()([tmp, c42])

        # Third block (up)
        tmp = BatchNormalization(axis=4, momentum=momentum, name=net_name + 'batch_norm_3.4', trainable=trainable)(end_42)
        tmp = Activation('relu')(tmp)
        tmp = UpSampling3D(size=pool_size, name=net_name + 'up_3')(tmp)
        tmp = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name=net_name + 'conv_up_3', padding='same', trainable=trainable)(tmp)
        tmp = Concatenate(axis=4)([tmp, end_31])
        tmp = BatchNormalization(axis=4, momentum=momentum, name=net_name + 'batch_norm_3.5', trainable=trainable)(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name=net_name + 'conv_3.3', padding='same', trainable=trainable)(tmp)
        tmp = BatchNormalization(axis=4, momentum=momentum, name=net_name + 'batch_norm_3.6', trainable=trainable)(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name=net_name + 'conv_3.4', padding='same', trainable=trainable)(tmp)

        c32 = UpSampling3D(size=pool_size, name=net_name + 'up_3conn')(end_42)
        c32 = Conv3D(32, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name=net_name + 'conv_conn_3.2', padding='same', trainable=trainable)(c32)

        end_32 = Add()([tmp, c32])

        # Second block (up)
        tmp = BatchNormalization(axis=4, momentum=momentum, name=net_name + 'batch_norm_2.4', trainable=trainable)(end_32)
        tmp = Activation('relu')(tmp)
        tmp = UpSampling3D(size=pool_size, name=net_name + 'up_2')(tmp)
        tmp = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name=net_name + 'conv_up_2', padding='same', trainable=trainable)(tmp)
        tmp = Concatenate(axis=4)([tmp, end_21])
        tmp = BatchNormalization(axis=4, momentum=momentum, name=net_name + 'batch_norm_2.5', trainable=trainable)(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name=net_name + 'conv_2.3', padding='same', trainable=trainable)(tmp)
        tmp = BatchNormalization(axis=4, momentum=momentum, name=net_name + 'batch_norm_2.6', trainable=trainable)(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name=net_name + 'conv_2.4', padding='same', trainable=trainable)(tmp)

        c22 = UpSampling3D(size=pool_size, name=net_name + 'up_2conn')(end_32)
        c22 = Conv3D(16, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name=net_name + 'conv_conn_2.2', padding='same', trainable=trainable)(c22)

        end_22 = Add()([tmp, c22])

        # First block (up)
        tmp = BatchNormalization(axis=4, momentum=momentum, name=net_name + 'batch_norm_1.4', trainable=trainable)(end_22)
        tmp = Activation('relu')(tmp)
        tmp = UpSampling3D(size=pool_size, name=net_name + 'up_1')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name=net_name + 'conv_up_1', padding='same', trainable=trainable)(tmp)
        tmp = Concatenate(axis=4)([tmp, end_11])
        tmp = BatchNormalization(axis=4, momentum=momentum, name=net_name + 'batch_norm_1.5', trainable=trainable)(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name=net_name + 'conv_1.3', padding='same', trainable=trainable)(tmp)
        tmp = BatchNormalization(axis=4, momentum=momentum, name=net_name + 'batch_norm_1.6', trainable=trainable)(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name=net_name + 'conv_1.4', padding='same', trainable=trainable)(tmp)

        c12 = UpSampling3D(size=pool_size, name=net_name + 'up_1_conn')(end_22)
        c12 = Conv3D(8, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name=net_name + 'conv_conn_1.2', padding='same', trainable=trainable)(c12)

        end_12 = Add()([tmp, c12])

        # Final convolution
        tmp = BatchNormalization(axis=4, momentum=momentum, name=net_name + 'batch_norm_1.7', trainable=trainable)(end_12)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name=net_name + 'conv_pre_softmax', padding='same', trainable=trainable)(tmp)
        tmp = BatchNormalization(axis=4, momentum=momentum, name=net_name + 'batch_norm_pre_softmax', trainable=trainable)(tmp)
        in_softmax = Activation('relu')(tmp)

        # Classification layer
        classification = Conv3D(num_classes, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                                name=net_name + 'final_convolution_1x1x1', trainable=trainable)(in_softmax)
        # classification = BatchNormalization(axis=4, name= net_name +'final_batch_norm')(classification)
        y = softmax_activation(classification)

        if survival:
            return y, end_5
        else:
            return y

    @staticmethod
    def vnet_2(num_modalities, segment_dimensions, num_classes, l1=0.0001, l2=0.001):
        input_shape_1 = segment_dimensions + (num_modalities,)
        input_shape_2 = segment_dimensions + (1,)
        input_shape_3 = segment_dimensions + (4,)
        output_shape = segment_dimensions

        initializer = 'he_normal'

        x1 = Input(shape=input_shape_1, name='V-net_input')
        x2 = Input(shape=input_shape_2, name='prismMask')


        classification, regression = BraTS_models.create_vnet(input=x1, num_classes=num_classes, net_name='net1_', trainable = False,
                                                 survival=True)

        # mask = MaxPooling3D(pool_size=(2,2,2))(x2)
        # mask = MaxPooling3D(pool_size=(2, 2, 2))(mask)
        # mask = MaxPooling3D(pool_size=(2, 2, 2))(mask)
        # mask = MaxPooling3D(pool_size=(2, 2, 2))(mask)
        # regression = Conv3D(1, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=l1_l2(l1=l1,l2=l2),
        #                     name='conv_reduced_features_survival', padding='same')(regression)
        # regression = Multiply()([regression, mask])
        # regression = BatchNormalization(axis=4, momentum=0.7, name='net_1_batch_norm_dense1')(regression)
        # regression = Activation('relu')(regression)
        # regression = Flatten()(regression)
        # regression = Dense(250,kernel_initializer='he_normal', kernel_regularizer=l1_l2(l1=l1,l2=l2))(regression)
        # regression = BatchNormalization(momentum=0.7)(regression)
        # regression = Dense(50, kernel_initializer='he_normal', kernel_regularizer=l1_l2(l1=l1, l2=l2))(regression)
        # regression = BatchNormalization(momentum=0.7)(regression)
        # regression = Dense(1, kernel_initializer='he_normal', kernel_regularizer=l1_l2(l1=l1, l2=l2))(regression)

        model = Model(inputs=[x1], outputs=[classification,regression])

        return model, output_shape

    @staticmethod
    def survival_net(num_modalities, segment_dimensions, num_classes, premodel, l1=0.0001, l2=0.001):
        image_input = Input(shape=segment_dimensions + (num_modalities,), name='SurvivalNet_input')
        mask_input = Input(shape=segment_dimensions + (1,), name='SurvivalNet_mask')

        _,regression = premodel(image_input)

        mask = MaxPooling3D(pool_size=(2,2,2))(mask_input)
        mask = MaxPooling3D(pool_size=(2, 2, 2))(mask)
        mask = MaxPooling3D(pool_size=(2, 2, 2))(mask)
        mask = MaxPooling3D(pool_size=(2, 2, 2))(mask)

        regression = Multiply()([regression, mask])
        regression = BatchNormalization(axis=4, momentum=0.7, name='net_1_batch_norm_dense1')(regression)
        regression = Activation('relu')(regression)
        regression = Conv3D(1, (1, 1, 1), kernel_initializer='he_normal', kernel_regularizer=l1_l2(l1=l1, l2=l2),
                            name='conv_reduced_features_survival', padding='same')(regression)
        regression = Flatten()(regression)
        regression = Dense(250,kernel_initializer='he_normal', kernel_regularizer=l1_l2(l1=l1,l2=l2))(regression)
        regression = BatchNormalization(momentum=0.7)(regression)
        regression = Dense(50, kernel_initializer='he_normal', kernel_regularizer=l1_l2(l1=l1, l2=l2))(regression)
        regression = BatchNormalization(momentum=0.7)(regression)
        regression = Dense(1, kernel_initializer='he_normal', kernel_regularizer=l1_l2(l1=l1, l2=l2))(regression)

        model = Model(inputs=[image_input,mask_input], outputs = regression)

        return model, (1,)
    @staticmethod
    def brats2017(num_modalities, segment_dimensions, num_classes, l1=0.0, l2=0.0):
        """
        U-Net based architecture for segmentation (http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)
            - Convolutional layers: 3x3x3 filters, stride 1, same border
            - Max pooling: 2x2x2
            - Upsampling layers: 2x2x2
            - Activations: ReLU
            - Classification layer: 1x1x1 kernel, and there are as many kernels as classes to be predicted. Element-wise
              softmax as activation.
            - Loss: 3D categorical cross-entropy
            - Optimizer: Adam with lr=0.001, beta_1=0.9, beta_2=0.999 and epsilon=10 ** (-4)
            - Regularization: L1=0.000001, L2=0.0001
            - Weights initialization: He et al normal initialization (https://arxiv.org/abs/1502.01852)

        Returns
        -------
        keras.model
            Compiled Sequential model
        tuple (dim1, dim2, dim3)
            Output shape computed from segment_dimensions and the convolutional architecture
        """
        if not isinstance(segment_dimensions, tuple) or len(segment_dimensions) != 3:
            raise ValueError('segment_dimensions must be a tuple with length 3, specifying the shape of the '
                             'input segment')

        # for dim in segment_dimensions:
        #     assert dim % 32 == 0  # As there are 5 (2, 2, 2) max-poolings, 2 ** 5 is the minimum input size

        # Hyperaparametre values

        initializer = 'he_normal'
        pool_size = (2, 2, 2)

        # Compute input shape, receptive field and output shape after softmax activation
        input_shape = segment_dimensions + (num_modalities,)
        output_shape = segment_dimensions

        # Activations, regularizers and optimizers
        # softmax_activation =
        regularizer = l1_l2(l1=l1, l2=l2)

        # Architecture definition
        # INPUT
        x = Input(shape=input_shape, name='V-net_input')

        first_conv = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer,
                            name='conv_initial', padding='same')(x)

        # First block (down)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.1')(first_conv)
        tmp = Activation('relu')(tmp)
        z1 = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.1',
                    padding='same')(tmp)

        c11 = Conv3D(8, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_conn_1.1',
                     padding='same')(x)
        end_11 = Add()([z1, c11])

        # Second block (down)
        tmp = BatchNormalization(axis=4, name='batch_norm_2.1')(end_11)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(16, (2, 2, 2), strides=(2, 2, 2), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='downpool_1', padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_2.2')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_2.1',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_2.3')(tmp)
        tmp = Activation('relu')(tmp)
        z2 = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_2.2',
                    padding='same')(tmp)

        c21 = MaxPooling3D(pool_size=pool_size, name='pool_1')(end_11)
        c21 = Conv3D(16, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_conn_2.1',
                     padding='same')(c21)

        end_21 = Add()([z2, c21])

        # Third block (down)
        tmp = BatchNormalization(axis=4, name='batch_norm_3.1')(end_21)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(32, (2, 2, 2), strides=(2, 2, 2), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='downpool_2', padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_3.2')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_3.1',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_3.3')(tmp)
        tmp = Activation('relu')(tmp)
        z3 = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_3.2',
                    padding='same')(tmp)

        c31 = MaxPooling3D(pool_size=pool_size, name='pool_2')(end_21)
        c31 = Conv3D(32, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_conn_3.1',
                     padding='same')(c31)

        end_31 = Add()([z3, c31])

        # Fourth block (down)
        tmp = BatchNormalization(axis=4, name='batch_norm_4.1')(end_31)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(64, (2, 2, 2), strides=(2, 2, 2), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='downpool_3', padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_4.2')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_4.1',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_4.3')(tmp)
        tmp = Activation('relu')(tmp)
        z4 = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_4.2',
                    padding='same')(tmp)

        c41 = MaxPooling3D(pool_size=pool_size, name='pool_3')(end_31)
        c41 = Conv3D(64, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_conn_4.1',
                     padding='same')(c41)

        end_41 = Add()([z4, c41])

        # Fifth block
        tmp = BatchNormalization(axis=4, name='batch_norm_5.1')(end_41)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(128, (2, 2, 2), strides=(2, 2, 2), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='downpool_4', padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_5.2')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(128, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_5.1',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_5.3')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(128, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_5.2',
                     padding='same')(tmp)  # inflection point

        c5 = MaxPooling3D(pool_size=pool_size, name='pool_4')(end_41)
        c5 = Conv3D(128, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_conn_5',
                    padding='same')(c5)

        end_5 = Add()([tmp, c5])

        # Fourth block (up)
        tmp = BatchNormalization(axis=4, name='batch_norm_4.4')(end_5)
        tmp = Activation('relu')(tmp)
        tmp = UpSampling3D(size=pool_size, name='up_4')(tmp)
        tmp = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_up_4',
                     padding='same')(tmp)
        tmp = Concatenate(axis=4)([tmp, end_41])
        tmp = BatchNormalization(axis=4, name='batch_norm_4.5')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_4.3',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_4.6')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_4.4',
                     padding='same')(tmp)

        c42 = UpSampling3D(size=pool_size, name='up_4conn')(end_5)
        c42 = Conv3D(64, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_conn_4.2',
                     padding='same')(c42)

        end_42 = Add()([tmp, c42])

        # Third block (up)
        tmp = BatchNormalization(axis=4, name='batch_norm_3.4')(end_42)
        tmp = Activation('relu')(tmp)
        tmp = UpSampling3D(size=pool_size, name='up_3')(tmp)
        tmp = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_up_3',
                     padding='same')(tmp)
        tmp = Concatenate(axis=4)([tmp, end_31])
        tmp = BatchNormalization(axis=4, name='batch_norm_3.5')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_3.3',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_3.6')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_3.4',
                     padding='same')(tmp)

        c32 = UpSampling3D(size=pool_size, name='up_3conn')(end_42)
        c32 = Conv3D(32, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_conn_3.2',
                     padding='same')(c32)

        end_32 = Add()([tmp, c32])

        # Second block (up)
        tmp = BatchNormalization(axis=4, name='batch_norm_2.4')(end_32)
        tmp = Activation('relu')(tmp)
        tmp = UpSampling3D(size=pool_size, name='up_2')(tmp)
        tmp = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_up_2',
                     padding='same')(tmp)
        tmp = Concatenate(axis=4)([tmp, end_21])
        tmp = BatchNormalization(axis=4, name='batch_norm_2.5')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_2.3',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_2.6')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_2.4',
                     padding='same')(tmp)

        c22 = UpSampling3D(size=pool_size, name='up_2conn')(end_32)
        c22 = Conv3D(16, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_conn_2.2',
                     padding='same')(c22)

        end_22 = Add()([tmp, c22])

        # First block (up)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.4')(end_22)
        tmp = Activation('relu')(tmp)
        tmp = UpSampling3D(size=pool_size, name='up_1')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_up_1',
                     padding='same')(tmp)
        tmp = Concatenate(axis=4)([tmp, end_11])
        tmp = BatchNormalization(axis=4, name='batch_norm_1.5')(tmp)
        end_root_net = Activation('relu')(tmp)

        c12 = UpSampling3D(size=pool_size, name='up_1_conn')(end_22)
        c12 = Conv3D(8, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_conn_1.2',
                     padding='same')(c12)

        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.3',
                     padding='same')(end_root_net)

        tmp = BatchNormalization(axis=4, name='batch_norm_1.6')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.4',
                     padding='same')(tmp)
        end_12 = Add()([tmp, c12])

        ############## Mask ################
        # Final convolution
        tmp = BatchNormalization(axis=4, name='batch_norm_1.7')(end_12)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_pre_softmax', padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_pre_softmax')(tmp)
        in_softmax = Activation('relu')(tmp)
        classification = Conv3D(2, (1, 1, 1), kernel_initializer=initializer,
                                name='final_convolution_1x1x1')(in_softmax)

        mask_brain = Input(shape=segment_dimensions + (1,), name='V-net_mask1')
        y = Lambda(elementwise_softmax_3d, name='Softmax')(classification)
        mask_brain1 = Concatenate()([mask_brain, mask_brain])
        mask_brain2 = Lambda(complementary_mask, output_shape=segment_dimensions + (1,),
                             name='complementary_brain_mask')(mask_brain)
        mask_brain2 = Concatenate()([mask_brain2, Lambda(lambda x: K.zeros_like(x))(mask_brain)])

        y = Multiply(name='mask_background')([y, mask_brain1])
        y_mask = Add(name='label_mask_background')([y, mask_brain2])

        y_mask_tumor = Lambda(lambda x: K.expand_dims(x[:, :, :, :, 1], axis=4))(y_mask)

        ############## Tumor ################
        # Final convolution
        tmp = BatchNormalization(axis=4, name='batch_norm_1.8')(end_12)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_pre_softmax_seg', padding='same')(tmp)

        tmp = BatchNormalization(axis=4, name='batch_norm_pre_softmax_seg')(tmp)
        in_softmax = Activation('relu')(tmp)

        classification = Conv3D(num_classes, (1, 1, 1), kernel_initializer=initializer,
                                name='final_convolution_1x1x1_seg')(in_softmax)

        y = Lambda(elementwise_softmax_3d, name='Softmax_seg')(classification)
        # mask_tumor = Lambda(repeat_channels(num_classes), output_shape=repeat_channels_shape(num_classes),
        #                       name='repeat_mask')(y_mask_tumor)
        true_mask_tumor = Input(shape=segment_dimensions + (1,))

        mask_tumor = Lambda(repeat_channels(num_classes), output_shape=repeat_channels_shape(num_classes),
                            name='repeat_mask')(true_mask_tumor)

        y = Multiply(name='mask_seg')([y, mask_tumor])
        # cmp_mask = Lambda(complementary_mask, output_shape = segment_dimensions + (1,),
        #                   name='complementary_tumor_mask')(y_mask_tumor)
        cmp_mask = Lambda(complementary_mask, output_shape=segment_dimensions + (1,),
                          name='complementary_tumor_mask')(true_mask_tumor)
        cmp_mask = Concatenate()([cmp_mask,
                                  Lambda(lambda x: K.zeros_like(x))(cmp_mask),
                                  Lambda(lambda x: K.zeros_like(x))(cmp_mask),
                                  Lambda(lambda x: K.zeros_like(x))(cmp_mask)])

        y_seg = Add(name='label_seg_background')([y, cmp_mask])

        labels_seg = Input(shape=segment_dimensions + (num_classes,), name='V-net_labels')
        loss_y_seg = Lambda(categorical_crossentropy_3d_masked, output_shape=(1,), name='masked_seg_loss')(
            [y_seg, y_mask_tumor, labels_seg])

        ############## Survival ################
        boolean_survival = Input(shape=(1,), name='V-net_boolean_survival')
        tmp = BatchNormalization(axis=4, name='batch_norm_1.9')(end_5)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(1, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_reduce_features', padding='same')(tmp)
        tmp = Dense(512)(Flatten()(tmp))
        y_survival = Dense(1)(tmp)
        y_survival = Multiply()([y_survival, boolean_survival])

        # Computation of semi-supervised loss
        model = Model(inputs=[x, mask_brain, labels_seg, true_mask_tumor, boolean_survival],
                      outputs=[y_mask, loss_y_seg, y_survival])

        return model, output_shape

    @staticmethod
    def brats2017_seg_only(num_modalities, segment_dimensions, num_classes, l1=0.0, l2=0.0):
        """
        U-Net based architecture for segmentation (http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)
            - Convolutional layers: 3x3x3 filters, stride 1, same border
            - Max pooling: 2x2x2
            - Upsampling layers: 2x2x2
            - Activations: ReLU
            - Classification layer: 1x1x1 kernel, and there are as many kernels as classes to be predicted. Element-wise
              softmax as activation.
            - Loss: 3D categorical cross-entropy
            - Optimizer: Adam with lr=0.001, beta_1=0.9, beta_2=0.999 and epsilon=10 ** (-4)
            - Regularization: L1=0.000001, L2=0.0001
            - Weights initialization: He et al normal initialization (https://arxiv.org/abs/1502.01852)

        Returns
        -------
        keras.model
            Compiled Sequential model
        tuple (dim1, dim2, dim3)
            Output shape computed from segment_dimensions and the convolutional architecture
        """
        if not isinstance(segment_dimensions, tuple) or len(segment_dimensions) != 3:
            raise ValueError('segment_dimensions must be a tuple with length 3, specifying the shape of the '
                             'input segment')

        # for dim in segment_dimensions:
        #     assert dim % 32 == 0  # As there are 5 (2, 2, 2) max-poolings, 2 ** 5 is the minimum input size

        # Hyperaparametre values

        initializer = 'he_normal'
        pool_size = (2, 2, 2)

        # Compute input shape, receptive field and output shape after softmax activation
        input_shape = segment_dimensions + (num_modalities,)
        output_shape = segment_dimensions

        # Activations, regularizers and optimizers
        # softmax_activation =
        regularizer = l1_l2(l1=l1, l2=l2)

        # Architecture definition
        # INPUT
        x = Input(shape=input_shape, name='V-net_input')

        first_conv = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer,
                            name='conv_initial', padding='same')(x)

        # First block (down)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.1')(first_conv)
        tmp = Activation('relu')(tmp)
        z1 = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.1',
                    padding='same')(tmp)

        c11 = Conv3D(8, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_conn_1.1',
                     padding='same')(x)
        end_11 = Add()([z1, c11])

        # Second block (down)
        tmp = BatchNormalization(axis=4, name='batch_norm_2.1')(end_11)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(16, (2, 2, 2), strides=(2, 2, 2), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='downpool_1', padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_2.2')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_2.1',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_2.3')(tmp)
        tmp = Activation('relu')(tmp)
        z2 = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_2.2',
                    padding='same')(tmp)

        c21 = MaxPooling3D(pool_size=pool_size, name='pool_1')(end_11)
        c21 = Conv3D(16, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_conn_2.1',
                     padding='same')(c21)

        end_21 = Add()([z2, c21])

        # Third block (down)
        tmp = BatchNormalization(axis=4, name='batch_norm_3.1')(end_21)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(32, (2, 2, 2), strides=(2, 2, 2), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='downpool_2', padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_3.2')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_3.1',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_3.3')(tmp)
        tmp = Activation('relu')(tmp)
        z3 = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_3.2',
                    padding='same')(tmp)

        c31 = MaxPooling3D(pool_size=pool_size, name='pool_2')(end_21)
        c31 = Conv3D(32, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_conn_3.1',
                     padding='same')(c31)

        end_31 = Add()([z3, c31])

        # Fourth block (down)
        tmp = BatchNormalization(axis=4, name='batch_norm_4.1')(end_31)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(64, (2, 2, 2), strides=(2, 2, 2), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='downpool_3', padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_4.2')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_4.1',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_4.3')(tmp)
        tmp = Activation('relu')(tmp)
        z4 = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_4.2',
                    padding='same')(tmp)

        c41 = MaxPooling3D(pool_size=pool_size, name='pool_3')(end_31)
        c41 = Conv3D(64, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_conn_4.1',
                     padding='same')(c41)

        end_41 = Add()([z4, c41])

        # Fifth block
        tmp = BatchNormalization(axis=4, name='batch_norm_5.1')(end_41)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(128, (2, 2, 2), strides=(2, 2, 2), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='downpool_4', padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_5.2')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(128, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_5.1',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_5.3')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(128, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_5.2',
                     padding='same')(tmp)  # inflection point

        c5 = MaxPooling3D(pool_size=pool_size, name='pool_4')(end_41)
        c5 = Conv3D(128, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_conn_5',
                    padding='same')(c5)

        end_5 = Add()([tmp, c5])

        # Fourth block (up)
        tmp = BatchNormalization(axis=4, name='batch_norm_4.4')(end_5)
        tmp = Activation('relu')(tmp)
        tmp = UpSampling3D(size=pool_size, name='up_4')(tmp)
        tmp = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_up_4',
                     padding='same')(tmp)
        tmp = Concatenate(axis=4)([tmp, end_41])
        tmp = BatchNormalization(axis=4, name='batch_norm_4.5')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_4.3',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_4.6')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_4.4',
                     padding='same')(tmp)

        c42 = UpSampling3D(size=pool_size, name='up_4conn')(end_5)
        c42 = Conv3D(64, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_conn_4.2',
                     padding='same')(c42)

        end_42 = Add()([tmp, c42])

        # Third block (up)
        tmp = BatchNormalization(axis=4, name='batch_norm_3.4')(end_42)
        tmp = Activation('relu')(tmp)
        tmp = UpSampling3D(size=pool_size, name='up_3')(tmp)
        tmp = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_up_3',
                     padding='same')(tmp)
        tmp = Concatenate(axis=4)([tmp, end_31])
        tmp = BatchNormalization(axis=4, name='batch_norm_3.5')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_3.3',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_3.6')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_3.4',
                     padding='same')(tmp)

        c32 = UpSampling3D(size=pool_size, name='up_3conn')(end_42)
        c32 = Conv3D(32, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_conn_3.2',
                     padding='same')(c32)

        end_32 = Add()([tmp, c32])

        # Second block (up)
        tmp = BatchNormalization(axis=4, name='batch_norm_2.4')(end_32)
        tmp = Activation('relu')(tmp)
        tmp = UpSampling3D(size=pool_size, name='up_2')(tmp)
        tmp = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_up_2',
                     padding='same')(tmp)
        tmp = Concatenate(axis=4)([tmp, end_21])
        tmp = BatchNormalization(axis=4, name='batch_norm_2.5')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_2.3',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_2.6')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_2.4',
                     padding='same')(tmp)

        c22 = UpSampling3D(size=pool_size, name='up_2conn')(end_32)
        c22 = Conv3D(16, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_conn_2.2',
                     padding='same')(c22)

        end_22 = Add()([tmp, c22])

        # First block (up)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.4')(end_22)
        tmp = Activation('relu')(tmp)
        tmp = UpSampling3D(size=pool_size, name='up_1')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_up_1',
                     padding='same')(tmp)
        tmp = Concatenate(axis=4)([tmp, end_11])
        tmp = BatchNormalization(axis=4, name='batch_norm_1.5')(tmp)
        end_root_net = Activation('relu')(tmp)

        c12 = UpSampling3D(size=pool_size, name='up_1_conn')(end_22)
        c12 = Conv3D(8, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_conn_1.2',
                     padding='same')(c12)

        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.3',
                     padding='same')(end_root_net)

        tmp = BatchNormalization(axis=4, name='batch_norm_1.6')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.4',
                     padding='same')(tmp)
        end_12 = Add()([tmp, c12])

        ############## Tumor ################
        # Final convolution
        tmp = BatchNormalization(axis=4, name='batch_norm_1.8')(end_12)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_pre_softmax_seg', padding='same')(tmp)

        tmp = BatchNormalization(axis=4, name='batch_norm_pre_softmax_seg')(tmp)
        in_softmax = Activation('relu')(tmp)

        classification = Conv3D(num_classes, (1, 1, 1), kernel_initializer=initializer,
                                name='final_convolution_1x1x1_seg')(in_softmax)

        y_softmax = Lambda(elementwise_softmax_3d, name='Softmax_seg')(classification)

        true_mask_tumor = Input(shape=segment_dimensions + (1,))

        mask_tumor = Lambda(repeat_channels(num_classes), output_shape=repeat_channels_shape(num_classes),
                            name='repeat_mask')(true_mask_tumor)

        cmp_mask = Lambda(complementary_mask, output_shape=segment_dimensions + (1,),
                          name='complementary_tumor_mask')(true_mask_tumor)
        # cmp_mask2 = Concatenate(axis=4)([cmp_mask,
        #                           Lambda(lambda x: K.zeros_like(x))(cmp_mask),
        #                           Lambda(lambda x: K.zeros_like(x))(cmp_mask),
        #                           Lambda(lambda x: K.zeros_like(x))(cmp_mask)])

        y = Multiply(name='mask_seg')([y_softmax, mask_tumor])
        y_seg = Concatenate(axis=4, name='label_seg_background')([
            cmp_mask,
            Lambda(lambda x: x[:, :, :, :, 1:2])(y),
            Lambda(lambda x: x[:, :, :, :, 2:3])(y),
            Lambda(lambda x: x[:, :, :, :, 3:])(y),
        ])
        # y_seg = Add(name='label_seg_background')([y, cmp_mask2])

        labels_seg = Input(shape=segment_dimensions + (num_classes,), name='V-net_labels')
        loss_y_seg = Lambda(categorical_crossentropy_3d_masked, output_shape=(1,), name='masked_seg_loss')(
            [y_seg, true_mask_tumor, labels_seg])


        # Computation of semi-supervised loss
        model = Model(inputs=[x, labels_seg, true_mask_tumor],
                      outputs=[loss_y_seg])

        return model, output_shape


    @staticmethod
    def brats2017_old(num_modalities, segment_dimensions, num_classes, l1=0.0, l2=0.0):
        """
        U-Net based architecture for segmentation (http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)
            - Convolutional layers: 3x3x3 filters, stride 1, same border
            - Max pooling: 2x2x2
            - Upsampling layers: 2x2x2
            - Activations: ReLU
            - Classification layer: 1x1x1 kernel, and there are as many kernels as classes to be predicted. Element-wise
              softmax as activation.
            - Loss: 3D categorical cross-entropy
            - Optimizer: Adam with lr=0.001, beta_1=0.9, beta_2=0.999 and epsilon=10 ** (-4)
            - Regularization: L1=0.000001, L2=0.0001
            - Weights initialization: He et al normal initialization (https://arxiv.org/abs/1502.01852)

        Returns
        -------
        keras.model
            Compiled Sequential model
        tuple (dim1, dim2, dim3)
            Output shape computed from segment_dimensions and the convolutional architecture
        """
        if not isinstance(segment_dimensions, tuple) or len(segment_dimensions) != 3:
            raise ValueError('segment_dimensions must be a tuple with length 3, specifying the shape of the '
                             'input segment')

        # for dim in segment_dimensions:
        #     assert dim % 32 == 0  # As there are 5 (2, 2, 2) max-poolings, 2 ** 5 is the minimum input size

        # Hyperaparametre values

        initializer = 'he_normal'
        pool_size = (2, 2, 2)

        # Compute input shape, receptive field and output shape after softmax activation
        input_shape = segment_dimensions + (num_modalities,)
        output_shape = segment_dimensions

        # Activations, regularizers and optimizers
        # softmax_activation =
        regularizer = l1_l2(l1=l1, l2=l2)

        # Architecture definition
        # INPUT
        x = Input(shape=input_shape, name='V-net_input')

        first_conv = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer,
                            name='conv_initial', padding='same')(x)

        # First block (down)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.1')(first_conv)
        tmp = Activation('relu')(tmp)
        z1 = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.1',
                    padding='same')(tmp)

        c11 = Conv3D(8, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_conn_1.1',
                     padding='same')(first_conv)
        end_11 = Add()([z1, c11])

        # Second block (down)
        tmp = BatchNormalization(axis=4, name='batch_norm_2.1')(end_11)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(16, (2, 2, 2), strides=(2, 2, 2), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='downpool_1', padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_2.2')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_2.1',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_2.3')(tmp)
        tmp = Activation('relu')(tmp)
        z2 = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_2.2',
                    padding='same')(tmp)

        c21 = MaxPooling3D(pool_size=pool_size, name='pool_1')(end_11)
        c21 = Conv3D(16, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_conn_2.1',
                     padding='same')(c21)

        end_21 = Add()([z2, c21])

        # Third block (down)
        tmp = BatchNormalization(axis=4, name='batch_norm_3.1')(end_21)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(32, (2, 2, 2), strides=(2, 2, 2), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='downpool_2', padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_3.2')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_3.1',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_3.3')(tmp)
        tmp = Activation('relu')(tmp)
        z3 = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_3.2',
                    padding='same')(tmp)

        c31 = MaxPooling3D(pool_size=pool_size, name='pool_2')(end_21)
        c31 = Conv3D(32, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_conn_3.1',
                     padding='same')(c31)

        end_31 = Add()([z3, c31])

        # Fourth block (down)
        tmp = BatchNormalization(axis=4, name='batch_norm_4.1')(end_31)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(64, (2, 2, 2), strides=(2, 2, 2), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='downpool_3', padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_4.2')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_4.1',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_4.3')(tmp)
        tmp = Activation('relu')(tmp)
        z4 = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_4.2',
                    padding='same')(tmp)

        c41 = MaxPooling3D(pool_size=pool_size, name='pool_3')(end_31)
        c41 = Conv3D(64, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_conn_4.1',
                     padding='same')(c41)

        end_41 = Add()([z4, c41])

        # Fifth block
        tmp = BatchNormalization(axis=4, name='batch_norm_5.1')(end_41)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(128, (2, 2, 2), strides=(2, 2, 2), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='downpool_4', padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_5.2')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(128, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_5.1',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_5.3')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(128, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_5.2',
                     padding='same')(tmp)  # inflection point

        c5 = MaxPooling3D(pool_size=pool_size, name='pool_4')(end_41)
        c5 = Conv3D(128, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_conn_5',
                    padding='same')(c5)

        end_5 = Add()([tmp, c5])

        # Fourth block (up)
        tmp = BatchNormalization(axis=4, name='batch_norm_4.4')(end_5)
        tmp = Activation('relu')(tmp)
        tmp = UpSampling3D(size=pool_size, name='up_4')(tmp)
        tmp = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_up_4',
                     padding='same')(tmp)
        tmp = Concatenate(axis=4)([tmp, z4])
        tmp = BatchNormalization(axis=4, name='batch_norm_4.5')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_4.3',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_4.6')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_4.4',
                     padding='same')(tmp)

        c42 = UpSampling3D(size=pool_size, name='up_4conn')(end_5)
        c42 = Conv3D(64, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_conn_4.2',
                     padding='same')(c42)

        end_42 = Add()([tmp, c42])

        # Third block (up)
        tmp = BatchNormalization(axis=4, name='batch_norm_3.4')(end_42)
        tmp = Activation('relu')(tmp)
        tmp = UpSampling3D(size=pool_size, name='up_3')(tmp)
        tmp = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_up_3',
                     padding='same')(tmp)
        tmp = Concatenate(axis=4)([tmp, z3])
        tmp = BatchNormalization(axis=4, name='batch_norm_3.5')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_3.3',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_3.6')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_3.4',
                     padding='same')(tmp)

        c32 = UpSampling3D(size=pool_size, name='up_3conn')(end_42)
        c32 = Conv3D(32, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_conn_3.2',
                     padding='same')(c32)

        end_32 = Add()([tmp, c32])

        # Second block (up)
        tmp = BatchNormalization(axis=4, name='batch_norm_2.4')(end_32)
        tmp = Activation('relu')(tmp)
        tmp = UpSampling3D(size=pool_size, name='up_2')(tmp)
        tmp = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_up_2',
                     padding='same')(tmp)
        tmp = Concatenate(axis=4)([tmp, z2])
        tmp = BatchNormalization(axis=4, name='batch_norm_2.5')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_2.3',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_2.6')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_2.4',
                     padding='same')(tmp)

        c22 = UpSampling3D(size=pool_size, name='up_2conn')(end_32)
        c22 = Conv3D(16, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_conn_2.2',
                     padding='same')(c22)

        end_22 = Add()([tmp, c22])

        # First block (up)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.4')(end_22)
        tmp = Activation('relu')(tmp)
        tmp = UpSampling3D(size=pool_size, name='up_1')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_up_1',
                     padding='same')(tmp)
        tmp = Concatenate(axis=4)([tmp, z1])
        tmp = BatchNormalization(axis=4, name='batch_norm_1.5')(tmp)
        end_root_net = Activation('relu')(tmp)

        c12 = UpSampling3D(size=pool_size, name='up_1_conn')(end_22)
        c12 = Conv3D(8, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_conn_1.2',
                     padding='same')(c12)

        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.3',
                     padding='same')(end_root_net)

        tmp = BatchNormalization(axis=4, name='batch_norm_1.6')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.4',
                     padding='same')(tmp)
        end_12 = Add()([tmp, c12])

        ############## Mask ################
        # Final convolution
        tmp = BatchNormalization(axis=4, name='batch_norm_1.7')(end_12)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_pre_softmax', padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_pre_softmax')(tmp)
        in_softmax = Activation('relu')(tmp)
        classification = Conv3D(2, (1, 1, 1), kernel_initializer=initializer,
                                name='final_convolution_1x1x1')(in_softmax)

        mask_brain = Input(shape=segment_dimensions + (1,), name='V-net_mask1')
        y = Lambda(elementwise_softmax_3d, name='Softmax')(classification)
        mask_brain1 = Concatenate()([mask_brain, mask_brain])
        mask_brain2 = Lambda(complementary_mask, output_shape=segment_dimensions + (1,),
                             name='complementary_brain_mask')(mask_brain)
        mask_brain2 = Concatenate()([mask_brain2, Lambda(lambda x: K.zeros_like(x))(mask_brain)])

        y = Multiply(name='mask_background')([y, mask_brain1])
        y_mask = Add(name='label_mask_background')([y, mask_brain2])

        y_mask_tumor = Lambda(lambda x: K.expand_dims(x[:, :, :, :, 1], axis=4))(y_mask)

        ############## Tumor ################
        # Final convolution
        tmp = BatchNormalization(axis=4, name='batch_norm_1.8')(end_12)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_pre_softmax_seg', padding='same')(tmp)

        tmp = BatchNormalization(axis=4, name='batch_norm_pre_softmax_seg')(tmp)
        in_softmax = Activation('relu')(tmp)

        classification = Conv3D(num_classes, (1, 1, 1), kernel_initializer=initializer,
                                name='final_convolution_1x1x1_seg')(in_softmax)

        y = Lambda(elementwise_softmax_3d, name='Softmax_seg')(classification)
        mask_tumor = Lambda(repeat_channels(num_classes), output_shape=repeat_channels_shape(num_classes),
                            name='repeat_mask')(y_mask_tumor)

        y = Multiply(name='mask_seg')([y, mask_tumor])
        cmp_mask = Lambda(complementary_mask, output_shape=segment_dimensions + (1,),
                          name='complementary_tumor_mask')(y_mask_tumor)
        cmp_mask = Concatenate()([cmp_mask,
                                  Lambda(lambda x: K.zeros_like(x))(cmp_mask),
                                  Lambda(lambda x: K.zeros_like(x))(cmp_mask),
                                  Lambda(lambda x: K.zeros_like(x))(cmp_mask)])

        y_seg = Add(name='label_seg_background')([y, cmp_mask])

        ############## Survival ################
        boolean_survival = Input(shape=(1,), name='V-net_boolean_survival')
        tmp = BatchNormalization(axis=4, name='batch_norm_1.9')(end_5)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(1, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_reduce_features', padding='same')(tmp)
        tmp = Dense(512)(Flatten()(tmp))
        y_survival = Dense(1)(tmp)
        y_survival = Multiply()([y_survival, boolean_survival])

        # Computation of semi-supervised loss
        model = Model(inputs=[x, mask_brain, boolean_survival], outputs=[y_mask, y_seg, y_survival])

        return model, output_shape


class iSeg_models(object):
    """ Interface that allows you to save and load models and their weights """

    # ------------------------------------------------ CONSTANTS ------------------------------------------------ #

    DEFAULT_MODEL = 'v_net'

    # ------------------------------------------------- METHODS ------------------------------------------------- #

    @classmethod
    def get_model(cls, num_modalities, segment_dimensions, num_classes, model_name=None,
                  **kwargs):
        """
        Returns the compiled model specified by its name.
        If no name is given, the default model is returned, which corresponds to the
        hand-picked model that performs the best.

        Parameters
        ----------
        num_modalities : int
            Number of modalities used as input channels
        segment_dimensions : tuple
            Tuple with 3 elements that specify the shape of the input segments
        num_classes : int
            Number of output classes to be predicted in the segmentation
        model_name : [Optional] String
            Name of the model to be returned
        weights_filename: [Optional] String
            Path to the H5 file containing the weights of the trained model. The user must ensure the
            weights correspond to the model to be loaded.

        Returns
        -------
        keras.Model
            Compiled keras model
        tuple
            Tuple with output size (necessary to adjust the ground truth matrix's size),
            or None if output size is the sames as segment_dimensions.

        Raises
        ------
        TypeError
            If model_name is not a String or None
        ValueError
            If the model specified by model_name does not exist
        """
        if model_name is None:
            return cls.get_model(num_modalities, segment_dimensions, num_classes, model_name=cls.DEFAULT_MODEL,
                                 **kwargs)
        if isinstance(model_name, str):
            try:

                model_getter = cls.__dict__[model_name]
                return model_getter.__func__(num_modalities, segment_dimensions, num_classes,
                                             **kwargs)
            except KeyError:
                raise ValueError('Model {} does not exist. Use the class method {} to know the available model names'
                                 .format(model_name, 'get_model_names'))
        else:
            raise TypeError('model_name must be a String/Unicode sequence or None')

    @classmethod
    def get_model_names(cls):
        """
        List of available models' names

        Returns
        -------
        List
            Names of the available model names to be used in get_model(model_name) method
        """

        def filter_functions(attr_name):
            tmp = ('model' not in attr_name) and ('MODEL' not in attr_name)
            return tmp and ('__' not in attr_name) and ('BASE' not in attr_name)

        return filter(filter_functions, cls.__dict__.keys())

    @staticmethod
    def compile(model, lr=0.0005, num_classes=2, loss_name='cross_entropy', optimizer_name='Adam'):
        if optimizer_name == 'Adam':
            beta_1 = 0.9
            beta_2 = 0.999
            epsilon = 10 ** (-8)
            optimizer = Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, clipnorm=1.)
        elif optimizer_name == 'RMSProp':
            rho = 0.9
            epsilon = 10 ** (-8)
            decay = 0.0
            optimizer = RMSprop(lr=lr, rho=rho, epsilon=epsilon, decay=decay)
        elif optimizer_name == 'SGD':
            optimizer = SGD(lr=lr, momentum=0.0, decay=0.0, nesterov=False)
        else:
            raise ValueError("Please, specify a valid optimizer when compiling")

        if num_classes == 2:
            metrics = [accuracy, dice_0, dice_1, recall_0, recall_1, precision_0, precision_1]
        elif num_classes == 3:
            metrics = [accuracy, dice_0, dice_1, dice_2, recall_0, recall_1, recall_2, precision_0, precision_1,
                       precision_2]
        elif num_classes == 4:
            metrics = [accuracy, dice_0, dice_1, dice_2, dice_3, recall_0, recall_1, recall_2, recall_3,
                       precision_0, precision_1, precision_2, precision_3]
        elif num_classes == 5:
            metrics = [accuracy, dice_whole, dice_core, dice_enhance, recall_0, recall_1, recall_2, recall_3, recall_4,
                       precision_0, precision_1, precision_2, precision_3, precision_4]
        else:
            raise ValueError("Please, specify a valid number of classes when compiling")

        if loss_name == 'cross_entropy':
            print('Loss: cross_entropy')
            loss = categorical_crossentropy_3d
        elif loss_name == 'dice':
            print('Loss: dice')
            loss = dice_cost
        else:
            raise ValueError('Please, specify a valid loss function')

        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics

        )

        return model


    @staticmethod
    def compile_ACNN(model, lr=0.0005 ):
        beta_1 = 0.9
        beta_2 = 0.999
        epsilon = 10 ** (-8)
        optimizer = Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, clipnorm=1.)

        loss = [lambda y_true, y_pred: y_pred, lambda y_true, y_pred: y_pred]

        model.compile(
            optimizer=optimizer,
            loss=loss,
            loss_weights = [1.0, 1.0]
        )
        return model


    @staticmethod
    def v_net_BN_masked_tmp(num_modalities, segment_dimensions, num_classes, shortcut_input=False,
                 l1=0.0, l2=0.0):
        """
        U-Net based architecture for segmentation (http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)
            - Convolutional layers: 3x3x3 filters, stride 1, same border
            - Max pooling: 2x2x2
            - Upsampling layers: 2x2x2
            - Activations: ReLU
            - Classification layer: 1x1x1 kernel, and there are as many kernels as classes to be predicted. Element-wise
              softmax as activation.
            - Loss: 3D categorical cross-entropy
            - Optimizer: Adam with lr=0.001, beta_1=0.9, beta_2=0.999 and epsilon=10 ** (-4)
            - Regularization: L1=0.000001, L2=0.0001
            - Weights initialization: He et al normal initialization (https://arxiv.org/abs/1502.01852)

        Returns
        -------
        keras.model
            Compiled Sequential model
        tuple (dim1, dim2, dim3)
            Output shape computed from segment_dimensions and the convolutional architecture
        """
        if not isinstance(segment_dimensions, tuple) or len(segment_dimensions) != 3:
            raise ValueError('segment_dimensions must be a tuple with length 3, specifying the shape of the '
                             'input segment')

        # for dim in segment_dimensions:
        #     assert dim % 32 == 0  # As there are 5 (2, 2, 2) max-poolings, 2 ** 5 is the minimum input size

        # Hyperaparametre values

        initializer = 'he_normal'
        pool_size = (2, 2, 2)

        # Compute input shape, receptive field and output shape after softmax activation
        input_shape = segment_dimensions + (num_modalities,)
        output_shape = segment_dimensions

        # Activations, regularizers and optimizers
        softmax_activation = Lambda(elementwise_softmax_3d, name='Softmax')
        regularizer = l1_l2(l1=l1, l2=l2)

        # Architecture definition
        # INPUT
        x = Input(shape=input_shape, name='V-net_input')
        true_mask = Input(shape=segment_dimensions + (1,), name='V-net_mask1')
        mask_0 = Lambda(repeat_channels(8), output_shape=repeat_channels_shape(8),
                           name='repeat_mask_8')(true_mask)

        # First block (down)
        first_conv = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer,
                            name='conv_initial', padding='same')(x)
        print(K.int_shape(mask_0))
        print(K.int_shape(first_conv))

        tmp = BatchNormalizationMasked(axis=4, name='batch_norm_1.1')([first_conv,mask_0])
        tmp = Activation('relu')(tmp)
        z1 = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.1',
                    padding='same')(tmp)

        c11 = Conv3D(8, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_conn_1.1',
                     padding='same')(x)
        end_11 = Add()([z1, c11])

        # First_b block (down)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.2_b')(end_11)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.2_b',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.3_b')(tmp)
        tmp = Activation('relu')(tmp)
        z1_b = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.3_b',
                      padding='same')(tmp)

        end_11_b = Add()([z1_b, end_11])

        # First_d block (down)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.2_c')(end_11_b)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.2_c',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.3_c')(tmp)
        tmp = Activation('relu')(tmp)
        z1_c = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.3_c',
                      padding='same')(tmp)

        end_11_c = Add()([z1_c, end_11_b])

        # First_c block (down)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.2_d')(end_11_c)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.2_d',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.3_d')(tmp)
        tmp = Activation('relu')(tmp)
        z1_d = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.3_d',
                      padding='same')(tmp)

        end_11_d = Add()([z1_d, end_11_c])

        # Second block (down)
        tmp = BatchNormalization(axis=4, name='batch_norm_2.1')(end_11)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(16, (2, 2, 2), strides=(2, 2, 2), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='downpool_1', padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_2.2')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_2.1',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_2.3')(tmp)
        tmp = Activation('relu')(tmp)
        z2 = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_2.2',
                    padding='same')(tmp)

        c21 = MaxPooling3D(pool_size=pool_size, name='pool_1')(end_11)
        c21 = Conv3D(16, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_conn_2.1', padding='same')(c21)

        end_21 = Add()([z2, c21])

        # Third block (down)
        tmp = BatchNormalization(axis=4, name='batch_norm_3.1')(end_21)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(32, (2, 2, 2), strides=(2, 2, 2), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='downpool_2', padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_3.2')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_3.1',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_3.3')(tmp)
        tmp = Activation('relu')(tmp)
        z3 = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_3.2',
                    padding='same')(tmp)

        c31 = MaxPooling3D(pool_size=pool_size, name='pool_2')(end_21)
        c31 = Conv3D(32, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_conn_3.1', padding='same')(c31)

        end_31 = Add()([z3, c31])

        # Fourth block (down)
        tmp = BatchNormalization(axis=4, name='batch_norm_4.1')(end_31)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(64, (2, 2, 2), strides=(2, 2, 2), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='downpool_3', padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_4.2')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_4.1',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_4.3')(tmp)
        tmp = Activation('relu')(tmp)
        z4 = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_4.2',
                    padding='same')(tmp)

        c41 = MaxPooling3D(pool_size=pool_size, name='pool_3')(end_31)
        c41 = Conv3D(64, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_conn_4.1', padding='same')(c41)

        end_41 = Add()([z4, c41])

        # Fifth block
        tmp = BatchNormalization(axis=4, name='batch_norm_5.1')(end_41)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(128, (2, 2, 2), strides=(2, 2, 2), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='downpool_4', padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_5.2')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(128, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_5.1',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_5.3')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(128, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_5.2',
                     padding='same')(tmp)  # inflection point

        c5 = MaxPooling3D(pool_size=pool_size, name='pool_4')(end_41)
        c5 = Conv3D(128, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_conn_5',
                    padding='same')(c5)

        end_5 = Add()([tmp, c5])

        # Fourth block (up)
        tmp = BatchNormalization(axis=4, name='batch_norm_4.4')(end_5)
        tmp = Activation('relu')(tmp)
        tmp = UpSampling3D(size=pool_size, name='up_4')(tmp)
        tmp = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_up_4',
                     padding='same')(tmp)
        tmp = Concatenate(axis=4)([tmp, z4])
        tmp = BatchNormalization(axis=4, name='batch_norm_4.5')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_4.3',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_4.6')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_4.4',
                     padding='same')(tmp)

        c42 = UpSampling3D(size=pool_size, name='up_4conn')(end_5)
        c42 = Conv3D(64, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_conn_4.2', padding='same')(c42)

        end_42 = Add()([tmp, c42])

        # Third block (up)
        tmp = BatchNormalization(axis=4, name='batch_norm_3.4')(end_42)
        tmp = Activation('relu')(tmp)
        tmp = UpSampling3D(size=pool_size, name='up_3')(tmp)
        tmp = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_up_3',
                     padding='same')(tmp)
        tmp = Concatenate(axis=4)([tmp, z3])
        tmp = BatchNormalization(axis=4, name='batch_norm_3.5')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_3.3',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_3.6')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_3.4',
                     padding='same')(tmp)

        c32 = UpSampling3D(size=pool_size, name='up_3conn')(end_42)
        c32 = Conv3D(32, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_conn_3.2', padding='same')(c32)

        end_32 = Add()([tmp, c32])

        # Second block (up)
        tmp = BatchNormalization(axis=4, name='batch_norm_2.4')(end_32)
        tmp = Activation('relu')(tmp)
        tmp = UpSampling3D(size=pool_size, name='up_2')(tmp)
        tmp = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_up_2',
                     padding='same')(tmp)
        tmp = Concatenate(axis=4)([tmp, z2])
        tmp = BatchNormalization(axis=4, name='batch_norm_2.5')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_2.3',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_2.6')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_2.4',
                     padding='same')(tmp)

        c22 = UpSampling3D(size=pool_size, name='up_2conn')(end_32)
        c22 = Conv3D(16, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_conn_2.2', padding='same')(c22)

        end_22 = Add()([tmp, c22])

        # First block (up)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.4')(end_22)
        tmp = Activation('relu')(tmp)
        tmp = UpSampling3D(size=pool_size, name='up_1')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_up_1',
                     padding='same')(tmp)
        tmp = Concatenate(axis=4)([tmp, z1])
        tmp = BatchNormalization(axis=4, name='batch_norm_1.5')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.3',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.6')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.4',
                     padding='same')(tmp)

        c12 = UpSampling3D(size=pool_size, name='up_1_conn')(end_22)
        c12 = Conv3D(8, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_conn_1.2',
                     padding='same')(c12)

        end_12 = Add()([tmp, c12])

        # Final convolution
        tmp = Concatenate(axis=4)([end_12, end_11_d])
        tmp = BatchNormalization(axis=4, name='batch_norm_1.7')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_pre_softmax', padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_pre_softmax')(tmp)
        in_softmax = Activation('relu')(tmp)

        # Classification layer
        if shortcut_input:
            in_softmax = Concatenate(axis=4)([in_softmax, x])

        classification = Conv3D(num_classes, (1, 1, 1), kernel_initializer=initializer,
                                name='final_convolution_1x1x1')(in_softmax)

        mask1 = Lambda(repeat_channels(num_classes), output_shape=repeat_channels_shape(num_classes),
                       name='repeat_mask')(true_mask)

        cmp_mask = Lambda(complementary_mask, output_shape=segment_dimensions + (1,),
                          name='complementary_tumor_mask')(true_mask)
        cmp_mask = Concatenate()([cmp_mask,
                                  Lambda(lambda x: K.zeros_like(x))(cmp_mask),
                                  Lambda(lambda x: K.zeros_like(x))(cmp_mask),
                                  Lambda(lambda x: K.zeros_like(x))(cmp_mask)])

        y = softmax_activation(classification)
        y = Multiply(name='mask_background')([y, mask1])
        y = Add(name='label_background')([y, cmp_mask])

        model = Model(inputs=[x, true_mask], outputs=y)

        return model, output_shape

    @staticmethod
    def v_net_BN(num_modalities, segment_dimensions, num_classes, shortcut_input=False,
              l1=0.0, l2=0.0):
        """
        U-Net based architecture for segmentation (http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)
            - Convolutional layers: 3x3x3 filters, stride 1, same border
            - Max pooling: 2x2x2
            - Upsampling layers: 2x2x2
            - Activations: ReLU
            - Classification layer: 1x1x1 kernel, and there are as many kernels as classes to be predicted. Element-wise
              softmax as activation.
            - Loss: 3D categorical cross-entropy
            - Optimizer: Adam with lr=0.001, beta_1=0.9, beta_2=0.999 and epsilon=10 ** (-4)
            - Regularization: L1=0.000001, L2=0.0001
            - Weights initialization: He et al normal initialization (https://arxiv.org/abs/1502.01852)

        Returns
        -------
        keras.model
            Compiled Sequential model
        tuple (dim1, dim2, dim3)
            Output shape computed from segment_dimensions and the convolutional architecture
        """
        if not isinstance(segment_dimensions, tuple) or len(segment_dimensions) != 3:
            raise ValueError('segment_dimensions must be a tuple with length 3, specifying the shape of the '
                             'input segment')

        # for dim in segment_dimensions:
        #     assert dim % 32 == 0  # As there are 5 (2, 2, 2) max-poolings, 2 ** 5 is the minimum input size

        # Hyperaparametre values

        initializer = 'he_normal'
        pool_size = (2, 2, 2)

        # Compute input shape, receptive field and output shape after softmax activation
        input_shape = segment_dimensions + (num_modalities,)
        output_shape = segment_dimensions

        # Activations, regularizers and optimizers
        softmax_activation = Lambda(elementwise_softmax_3d, name='Softmax')
        regularizer = l1_l2(l1=l1, l2=l2)

        # Architecture definition
        # INPUT
        x = Input(shape=input_shape, name='V-net_input')


        # First block (down)
        first_conv = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer,
                            name='conv_initial', padding='same')(x)
        tmp = BatchNormalizationMasked(axis=4, name='batch_norm_1.1')(first_conv)
        tmp = Activation('relu')(tmp)
        z1 = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.1',
                    padding='same')(tmp)

        c11 = Conv3D(8, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_conn_1.1',
                     padding='same')(x)
        end_11 = Add()([z1, c11])






        # First_b block (down)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.2_b')(end_11)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.2_b',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.3_b')(tmp)
        tmp = Activation('relu')(tmp)
        z1_b = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.3_b',
                    padding='same')(tmp)


        end_11_b = Add()([z1_b, end_11])




        # First_d block (down)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.2_c')(end_11_b)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.2_c',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.3_c')(tmp)
        tmp = Activation('relu')(tmp)
        z1_c = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.3_c',
                    padding='same')(tmp)


        end_11_c = Add()([z1_c, end_11_b])



        # First_c block (down)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.2_d')(end_11_c)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.2_d',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.3_d')(tmp)
        tmp = Activation('relu')(tmp)
        z1_d = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.3_d',
                    padding='same')(tmp)

        end_11_d = Add()([z1_d, end_11_c])






        # Second block (down)
        tmp = BatchNormalization(axis=4, name='batch_norm_2.1')(end_11)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(16, (2, 2, 2), strides=(2, 2, 2), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='downpool_1', padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_2.2')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_2.1',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_2.3')(tmp)
        tmp = Activation('relu')(tmp)
        z2 = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_2.2',
                    padding='same')(tmp)

        c21 = MaxPooling3D(pool_size=pool_size, name='pool_1')(end_11)
        c21 = Conv3D(16, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_conn_2.1', padding='same')(c21)

        end_21 = Add()([z2, c21])

        # Third block (down)
        tmp = BatchNormalization(axis=4, name='batch_norm_3.1')(end_21)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(32, (2, 2, 2), strides=(2, 2, 2), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='downpool_2', padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_3.2')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_3.1',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_3.3')(tmp)
        tmp = Activation('relu')(tmp)
        z3 = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_3.2',
                    padding='same')(tmp)

        c31 = MaxPooling3D(pool_size=pool_size, name='pool_2')(end_21)
        c31 = Conv3D(32, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_conn_3.1', padding='same')(c31)

        end_31 = Add()([z3, c31])

        # Fourth block (down)
        tmp = BatchNormalization(axis=4, name='batch_norm_4.1')(end_31)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(64, (2, 2, 2), strides=(2, 2, 2), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='downpool_3', padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_4.2')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_4.1',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_4.3')(tmp)
        tmp = Activation('relu')(tmp)
        z4 = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_4.2',
                    padding='same')(tmp)

        c41 = MaxPooling3D(pool_size=pool_size, name='pool_3')(end_31)
        c41 = Conv3D(64, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_conn_4.1', padding='same')(c41)

        end_41 = Add()([z4, c41])

        # Fifth block
        tmp = BatchNormalization(axis=4, name='batch_norm_5.1')(end_41)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(128, (2, 2, 2), strides=(2, 2, 2), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='downpool_4', padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_5.2')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(128, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_5.1',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_5.3')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(128, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_5.2',
                     padding='same')(tmp)  # inflection point

        c5 = MaxPooling3D(pool_size=pool_size, name='pool_4')(end_41)
        c5 = Conv3D(128, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_conn_5',
                    padding='same')(c5)

        end_5 = Add()([tmp, c5])

        # Fourth block (up)
        tmp = BatchNormalization(axis=4, name='batch_norm_4.4')(end_5)
        tmp = Activation('relu')(tmp)
        tmp = UpSampling3D(size=pool_size, name='up_4')(tmp)
        tmp = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_up_4',
                     padding='same')(tmp)
        tmp = Concatenate(axis=4)([tmp, z4])
        tmp = BatchNormalization(axis=4, name='batch_norm_4.5')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_4.3',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_4.6')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_4.4',
                     padding='same')(tmp)

        c42 = UpSampling3D(size=pool_size, name='up_4conn')(end_5)
        c42 = Conv3D(64, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_conn_4.2', padding='same')(c42)

        end_42 = Add()([tmp, c42])

        # Third block (up)
        tmp = BatchNormalization(axis=4, name='batch_norm_3.4')(end_42)
        tmp = Activation('relu')(tmp)
        tmp = UpSampling3D(size=pool_size, name='up_3')(tmp)
        tmp = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_up_3',
                     padding='same')(tmp)
        tmp = Concatenate(axis=4)([tmp, z3])
        tmp = BatchNormalization(axis=4, name='batch_norm_3.5')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_3.3',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_3.6')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_3.4',
                     padding='same')(tmp)

        c32 = UpSampling3D(size=pool_size, name='up_3conn')(end_42)
        c32 = Conv3D(32, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_conn_3.2', padding='same')(c32)

        end_32 = Add()([tmp, c32])

        # Second block (up)
        tmp = BatchNormalization(axis=4, name='batch_norm_2.4')(end_32)
        tmp = Activation('relu')(tmp)
        tmp = UpSampling3D(size=pool_size, name='up_2')(tmp)
        tmp = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_up_2',
                     padding='same')(tmp)
        tmp = Concatenate(axis=4)([tmp, z2])
        tmp = BatchNormalization(axis=4, name='batch_norm_2.5')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_2.3',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_2.6')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_2.4',
                     padding='same')(tmp)

        c22 = UpSampling3D(size=pool_size, name='up_2conn')(end_32)
        c22 = Conv3D(16, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_conn_2.2', padding='same')(c22)

        end_22 = Add()([tmp, c22])

        # First block (up)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.4')(end_22)
        tmp = Activation('relu')(tmp)
        tmp = UpSampling3D(size=pool_size, name='up_1')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_up_1',
                     padding='same')(tmp)
        tmp = Concatenate(axis=4)([tmp, z1])
        tmp = BatchNormalization(axis=4, name='batch_norm_1.5')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.3',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.6')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.4',
                     padding='same')(tmp)

        c12 = UpSampling3D(size=pool_size, name='up_1_conn')(end_22)
        c12 = Conv3D(8, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_conn_1.2',
                     padding='same')(c12)

        end_12 = Add()([tmp, c12])

        # Final convolution
        tmp = Concatenate(axis=4)([end_12, end_11_d])
        tmp = BatchNormalization(axis=4, name='batch_norm_1.7')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_pre_softmax', padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_pre_softmax')(tmp)
        in_softmax = Activation('relu')(tmp)

        # Classification layer
        if shortcut_input:
            in_softmax = Concatenate(axis=4)([in_softmax, x])

        classification = Conv3D(num_classes, (1, 1, 1), kernel_initializer=initializer,
                                name='final_convolution_1x1x1')(in_softmax)


        true_mask = Input(shape=segment_dimensions + (1,), name='V-net_mask1')
        mask1 = Lambda(repeat_channels(num_classes), output_shape=repeat_channels_shape(num_classes),
                       name='repeat_mask')(true_mask)

        cmp_mask = Lambda(complementary_mask, output_shape=segment_dimensions + (1,),
                          name='complementary_tumor_mask')(true_mask)
        cmp_mask = Concatenate()([cmp_mask,
                                  Lambda(lambda x: K.zeros_like(x))(cmp_mask),
                                  Lambda(lambda x: K.zeros_like(x))(cmp_mask),
                                  Lambda(lambda x: K.zeros_like(x))(cmp_mask)])

        y = softmax_activation(classification)
        y = Multiply(name='mask_background')([y, mask1])
        y = Add(name='label_background')([y, cmp_mask])

        model = Model(inputs=[x, true_mask], outputs=y)

        return model, output_shape

    @staticmethod
    def v_net_BN_masked(num_modalities, segment_dimensions, num_classes, shortcut_input=False,
              l1=0.0, l2=0.0):
        """
        U-Net based architecture for segmentation (http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)
            - Convolutional layers: 3x3x3 filters, stride 1, same border
            - Max pooling: 2x2x2
            - Upsampling layers: 2x2x2
            - Activations: ReLU
            - Classification layer: 1x1x1 kernel, and there are as many kernels as classes to be predicted. Element-wise
              softmax as activation.
            - Loss: 3D categorical cross-entropy
            - Optimizer: Adam with lr=0.001, beta_1=0.9, beta_2=0.999 and epsilon=10 ** (-4)
            - Regularization: L1=0.000001, L2=0.0001
            - Weights initialization: He et al normal initialization (https://arxiv.org/abs/1502.01852)

        Returns
        -------
        keras.model
            Compiled Sequential model
        tuple (dim1, dim2, dim3)
            Output shape computed from segment_dimensions and the convolutional architecture
        """
        if not isinstance(segment_dimensions, tuple) or len(segment_dimensions) != 3:
            raise ValueError('segment_dimensions must be a tuple with length 3, specifying the shape of the '
                             'input segment')

        # for dim in segment_dimensions:
        #     assert dim % 32 == 0  # As there are 5 (2, 2, 2) max-poolings, 2 ** 5 is the minimum input size

        # Hyperaparametre values

        initializer = 'he_normal'
        pool_size = (2, 2, 2)
        momentum = 0.7
        # Compute input shape, receptive field and output shape after softmax activation
        input_shape = segment_dimensions + (num_modalities,)
        output_shape = segment_dimensions

        # Activations, regularizers and optimizers
        softmax_activation = Lambda(elementwise_softmax_3d, name='Softmax')
        regularizer = l1_l2(l1=l1, l2=l2)

        # Architecture definition
        # INPUT
        x = Input(shape=input_shape, name='V-net_input')
        true_mask = Input(shape=segment_dimensions + (1,), name='V-net_mask1')
        true_mask_0 = Lambda(repeat_channels(8), output_shape=repeat_channels_shape(8),
                       name='repeat_mask_8')(true_mask)
        true_mask_1 = MaxPooling3D(pool_size=pool_size)(true_mask)
        true_mask_1 = Lambda(repeat_channels(16), output_shape=repeat_channels_shape(16),
                           name='repeat_mask_16')(true_mask_1)
        true_mask_2 = MaxPooling3D(pool_size=pool_size)(true_mask)
        true_mask_2 = Lambda(repeat_channels(32), output_shape=repeat_channels_shape(32),
                           name='repeat_mask_32')(true_mask_2)
        true_mask_3 = MaxPooling3D(pool_size=pool_size)(true_mask)
        true_mask_3 = Lambda(repeat_channels(64), output_shape=repeat_channels_shape(64),
                           name='repeat_mask_64')(true_mask_3)
        true_mask_4 = MaxPooling3D(pool_size=pool_size)(true_mask)
        true_mask_4 = Lambda(repeat_channels(128), output_shape=repeat_channels_shape(128),
                           name='repeat_mask_128')(true_mask_4)
        # true_mask_0 = UpSampling3D(size=pool_size)(true_mask)

        # First block (down)
        first_conv = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer,
                            name='conv_initial', padding='same')(x)
        tmp = BatchNormalizationMasked(momentum=momentum,axis=4, name='batch_norm_1.1')([first_conv,true_mask_0])
        tmp = Activation('relu')(tmp)
        z1 = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.1',
                    padding='same')(tmp)

        c11 = Conv3D(8, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_conn_1.1',
                     padding='same')(x)
        end_11 = Add()([z1, c11])






        # First_b block (down)
        # tmp = BatchNormalizationMasked(momentum=momentum,axis=4, name='batch_norm_1.2_b')(x_up,true_mask_0)
        # tmp = Activation('relu')(tmp)

        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.2_b',
                     padding='same')(x)
        tmp = BatchNormalizationMasked(momentum=momentum,axis=4, name='batch_norm_1.3_b')([tmp,true_mask_0])
        tmp = Activation('relu')(tmp)
        z1_b = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.3_b',
                    padding='same')(tmp)

        tmp = Conv3D(8, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.4_b',
                    padding='same')(x)
        end_11_b = Add()([z1_b, tmp])




        # First_d block (down)
        tmp = BatchNormalizationMasked(momentum=momentum,axis=4, name='batch_norm_1.2_c')([end_11_b,true_mask_0])
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.2_c',
                     padding='same')(tmp)
        tmp = BatchNormalizationMasked(momentum=momentum,axis=4, name='batch_norm_1.3_c')([tmp,true_mask_0])
        tmp = Activation('relu')(tmp)
        z1_c = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.3_c',
                    padding='same')(tmp)


        end_11_c = Add()([z1_c, end_11_b])



        # First_c block (down)
        tmp = BatchNormalizationMasked(momentum=momentum,axis=4, name='batch_norm_1.2_d')([end_11_c,true_mask_0])
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.2_d',
                     padding='same')(tmp)
        tmp = BatchNormalizationMasked(momentum=momentum,axis=4, name='batch_norm_1.3_d')([tmp,true_mask_0])
        tmp = Activation('relu')(tmp)
        z1_d = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.3_d',
                    padding='same')(tmp)

        end_11_d = Add()([z1_d, end_11_c])






        # Second block (down)
        tmp = BatchNormalizationMasked(momentum=momentum,axis=4, name='batch_norm_2.1')([end_11,true_mask_0])
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(16, (2, 2, 2), strides=(2, 2, 2), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='downpool_1', padding='same')(tmp)
        tmp = BatchNormalizationMasked(momentum=momentum,axis=4, name='batch_norm_2.2')([tmp, true_mask_1])
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_2.1',
                     padding='same')(tmp)
        tmp = BatchNormalizationMasked(momentum=momentum,axis=4, name='batch_norm_2.3')([tmp, true_mask_1])
        tmp = Activation('relu')(tmp)
        z2 = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_2.2',
                    padding='same')(tmp)

        c21 = MaxPooling3D(pool_size=pool_size, name='pool_1')(end_11)
        c21 = Conv3D(16, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_conn_2.1', padding='same')(c21)

        end_21 = Add()([z2, c21])

        # Third block (down)
        tmp = BatchNormalizationMasked(momentum=momentum,axis=4, name='batch_norm_3.1')([end_21, true_mask_1])
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(32, (2, 2, 2), strides=(2, 2, 2), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='downpool_2', padding='same')(tmp)
        tmp = BatchNormalizationMasked(momentum=momentum,axis=4, name='batch_norm_3.2')([tmp, true_mask_2])
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_3.1',
                     padding='same')(tmp)
        tmp = BatchNormalizationMasked(momentum=momentum,axis=4, name='batch_norm_3.3')([tmp, true_mask_2])
        tmp = Activation('relu')(tmp)
        z3 = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_3.2',
                    padding='same')(tmp)

        c31 = MaxPooling3D(pool_size=pool_size, name='pool_2')(end_21)
        c31 = Conv3D(32, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_conn_3.1', padding='same')(c31)

        end_31 = Add()([z3, c31])

        # Fourth block (down)
        tmp = BatchNormalizationMasked(momentum=momentum,axis=4, name='batch_norm_4.1')([end_31, true_mask_2])
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(64, (2, 2, 2), strides=(2, 2, 2), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='downpool_3', padding='same')(tmp)
        tmp = BatchNormalizationMasked(momentum=momentum,axis=4, name='batch_norm_4.2')([tmp, true_mask_3])
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_4.1',
                     padding='same')(tmp)
        tmp = BatchNormalizationMasked(momentum=momentum,axis=4, name='batch_norm_4.3')([tmp, true_mask_3])
        tmp = Activation('relu')(tmp)
        z4 = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_4.2',
                    padding='same')(tmp)

        c41 = MaxPooling3D(pool_size=pool_size, name='pool_3')(end_31)
        c41 = Conv3D(64, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_conn_4.1', padding='same')(c41)

        end_41 = Add()([z4, c41])

        # Fifth block
        tmp = BatchNormalizationMasked(momentum=momentum,axis=4, name='batch_norm_5.1')([end_41, true_mask_3])
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(128, (2, 2, 2), strides=(2, 2, 2), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='downpool_4', padding='same')(tmp)
        tmp = BatchNormalizationMasked(momentum=momentum,axis=4, name='batch_norm_5.2')([tmp, true_mask_4])
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(128, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_5.1',
                     padding='same')(tmp)
        tmp = BatchNormalizationMasked(momentum=momentum,axis=4, name='batch_norm_5.3')([tmp, true_mask_4])
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(128, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_5.2',
                     padding='same')(tmp)  # inflection point

        c5 = MaxPooling3D(pool_size=pool_size, name='pool_4')(end_41)
        c5 = Conv3D(128, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_conn_5',
                    padding='same')(c5)

        end_5 = Add()([tmp, c5])

        # Fourth block (up)
        tmp = BatchNormalizationMasked(momentum=momentum,axis=4, name='batch_norm_4.4')([end_5, true_mask_4])
        tmp = Activation('relu')(tmp)
        tmp = UpSampling3D(size=pool_size, name='up_4')(tmp)
        tmp = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_up_4',
                     padding='same')(tmp)
        tmp = Concatenate(axis=4)([tmp, z4])
        true_mask_3_2 = Lambda(repeat_channels(2), output_shape=repeat_channels_shape(2),
                               name='repeat_mask_64_2')(true_mask_3)
        tmp = BatchNormalizationMasked(momentum=momentum,axis=4, name='batch_norm_4.5')([tmp, true_mask_3_2])
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_4.3',
                     padding='same')(tmp)
        tmp = BatchNormalizationMasked(momentum=momentum,axis=4, name='batch_norm_4.6')([tmp, true_mask_3])
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_4.4',
                     padding='same')(tmp)

        c42 = UpSampling3D(size=pool_size, name='up_4conn')(end_5)
        c42 = Conv3D(64, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_conn_4.2', padding='same')(c42)

        end_42 = Add()([tmp, c42])

        # Third block (up)
        tmp = BatchNormalizationMasked(momentum=momentum,axis=4, name='batch_norm_3.4')([end_42, true_mask_3])
        tmp = Activation('relu')(tmp)
        tmp = UpSampling3D(size=pool_size, name='up_3')(tmp)
        tmp = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_up_3',
                     padding='same')(tmp)
        tmp = Concatenate(axis=4)([tmp, z3])
        true_mask_2_2 = Lambda(repeat_channels(2), output_shape=repeat_channels_shape(2),
                               name='repeat_mask_2_2')(true_mask_2)
        tmp = BatchNormalizationMasked(momentum=momentum,axis=4, name='batch_norm_3.5')([tmp,true_mask_2_2])
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_3.3',
                     padding='same')(tmp)
        tmp = BatchNormalizationMasked(momentum=momentum,axis=4, name='batch_norm_3.6')([tmp,true_mask_2])
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_3.4',
                     padding='same')(tmp)

        c32 = UpSampling3D(size=pool_size, name='up_3conn')(end_42)
        c32 = Conv3D(32, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_conn_3.2', padding='same')(c32)

        end_32 = Add()([tmp, c32])

        # Second block (up)
        tmp = BatchNormalizationMasked(momentum=momentum,axis=4, name='batch_norm_2.4')([end_32, true_mask_2])
        tmp = Activation('relu')(tmp)
        tmp = UpSampling3D(size=pool_size, name='up_2')(tmp)
        tmp = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_up_2',
                     padding='same')(tmp)
        tmp = Concatenate(axis=4)([tmp, z2])
        true_mask_1_2 = Lambda(repeat_channels(2), output_shape=repeat_channels_shape(2),
                               name='repeat_mask_1_2')(true_mask_1)
        tmp = BatchNormalizationMasked(momentum=momentum,axis=4, name='batch_norm_2.5')([tmp, true_mask_1_2])
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_2.3',
                     padding='same')(tmp)
        tmp = BatchNormalizationMasked(momentum=momentum,axis=4, name='batch_norm_2.6')([tmp, true_mask_1])
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_2.4',
                     padding='same')(tmp)

        c22 = UpSampling3D(size=pool_size, name='up_2conn')(end_32)
        c22 = Conv3D(16, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_conn_2.2', padding='same')(c22)

        end_22 = Add()([tmp, c22])

        # First block (up)
        tmp = BatchNormalizationMasked(momentum=momentum,axis=4, name='batch_norm_1.4')([end_22, true_mask_1])
        tmp = Activation('relu')(tmp)
        tmp = UpSampling3D(size=pool_size, name='up_1')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_up_1',
                     padding='same')(tmp)
        tmp = Concatenate(axis=4)([tmp, z1])
        true_mask_0_2 = Lambda(repeat_channels(2), output_shape=repeat_channels_shape(2),
                               name='repeat_mask_0_2')(true_mask_0)
        tmp = BatchNormalizationMasked(momentum=momentum,axis=4, name='batch_norm_1.5')([tmp, true_mask_0_2])
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.3',
                     padding='same')(tmp)
        tmp = BatchNormalizationMasked(momentum=momentum,axis=4, name='batch_norm_1.6')([tmp, true_mask_0])
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.4',
                     padding='same')(tmp)

        c12 = UpSampling3D(size=pool_size, name='up_1_conn')(end_22)
        c12 = Conv3D(8, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_conn_1.2',
                     padding='same')(c12)

        end_12 = Add()([tmp, c12])

        # Final convolution
        # end_11_d = MaxPooling3D(pool_size=pool_size, name='sr_down')(end_11_d)
        tmp = Concatenate(axis=4)([end_12, end_11_d])
        # tmp=end_12
        tmp = BatchNormalizationMasked(momentum=momentum,axis=4, name='batch_norm_1.7')([tmp, true_mask_0_2])
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_pre_softmax', padding='same')(tmp)
        tmp = BatchNormalizationMasked(momentum=momentum,axis=4, name='batch_norm_pre_softmax')([tmp, true_mask_0])
        in_softmax = Activation('relu')(tmp)

        # Classification layer
        if shortcut_input:
            in_softmax = Concatenate(axis=4)([in_softmax, x])

        classification = Conv3D(num_classes, (1, 1, 1), kernel_initializer=initializer,
                                name='final_convolution_1x1x1')(in_softmax)


        mask_pred = Lambda(repeat_channels(num_classes), output_shape=repeat_channels_shape(num_classes),
                       name='repeat_mask')(true_mask)

        cmp_mask = Lambda(complementary_mask, output_shape=segment_dimensions + (1,),
                          name='complementary_tumor_mask')(true_mask)
        cmp_mask = Concatenate()([cmp_mask,
                                  Lambda(lambda x: K.zeros_like(x))(cmp_mask),
                                  Lambda(lambda x: K.zeros_like(x))(cmp_mask),
                                  Lambda(lambda x: K.zeros_like(x))(cmp_mask)])

        y = softmax_activation(classification)
        y = Multiply(name='mask_background')([y, mask_pred])
        y = Add(name='label_background')([y, cmp_mask])

        model = Model(inputs=[x, true_mask], outputs=y)

        return model, output_shape



    @staticmethod
    def v_net(num_modalities, segment_dimensions, num_classes, l1=0.0, l2=0.0):
        """
        U-Net based architecture for segmentation (http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)
            - Convolutional layers: 3x3x3 filters, stride 1, same border
            - Max pooling: 2x2x2
            - Upsampling layers: 2x2x2
            - Activations: ReLU
            - Classification layer: 1x1x1 kernel, and there are as many kernels as classes to be predicted. Element-wise
              softmax as activation.
            - Loss: 3D categorical cross-entropy
            - Optimizer: Adam with lr=0.001, beta_1=0.9, beta_2=0.999 and epsilon=10 ** (-4)
            - Regularization: L1=0.000001, L2=0.0001
            - Weights initialization: He et al normal initialization (https://arxiv.org/abs/1502.01852)

        Returns
        -------
        keras.model
            Compiled Sequential model
        tuple (dim1, dim2, dim3)
            Output shape computed from segment_dimensions and the convolutional architecture
        """
        if not isinstance(segment_dimensions, tuple) or len(segment_dimensions) != 3:
            raise ValueError('segment_dimensions must be a tuple with length 3, specifying the shape of the '
                             'input segment')

        # for dim in segment_dimensions:
        #     assert dim % 32 == 0  # As there are 5 (2, 2, 2) max-poolings, 2 ** 5 is the minimum input size

        # Hyperaparametre values

        initializer = 'he_normal'
        pool_size = (2, 2, 2)

        # Compute input shape, receptive field and output shape after softmax activation
        input_shape = segment_dimensions + (num_modalities,)
        output_shape = segment_dimensions

        # Activations, regularizers and optimizers
        softmax_activation = Lambda(elementwise_softmax_3d, name='Softmax')
        regularizer = l1_l2(l1=l1, l2=l2)

        # Architecture definition
        # INPUT
        x = Input(shape=input_shape, name='V-net_input')


        # First block (down)
        first_conv = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer,
                            name='conv_initial', padding='same')(x)
        tmp = Activation('relu')(first_conv)
        z1 = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.1',
                    padding='same')(tmp)

        c11 = Conv3D(8, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_conn_1.1',
                     padding='same')(x)
        end_11 = Add()([z1, c11])


        # First_b block (down)
        tmp = Activation('relu')(end_11)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_2.1_b',
                     padding='same')(tmp)
        tmp = Activation('relu')(tmp)
        z1_b = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_2.2_b',
                    padding='same')(tmp)


        end_11_b = Add()([z1_b, end_11])




        # First_d block (down)
        tmp = Activation('relu')(end_11_b)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_2.1_c',
                     padding='same')(tmp)
        tmp = Activation('relu')(tmp)
        z1_c = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_2.2_c',
                    padding='same')(tmp)


        end_11_c = Add()([z1_c, end_11_b])



        # First_c block (down)
        tmp = Activation('relu')(end_11_c)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_2.1_d',
                     padding='same')(tmp)
        tmp = Activation('relu')(tmp)
        z1_d = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_2.2_d',
                    padding='same')(tmp)

        end_11_d = Add()([z1_d, end_11_c])






        # Second block (down)
        tmp = Activation('relu')(end_11)
        tmp = Conv3D(16, (2, 2, 2), strides=(2, 2, 2), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='downpool_1', padding='same')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_2.1',
                     padding='same')(tmp)
        tmp = Activation('relu')(tmp)
        z2 = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_2.2',
                    padding='same')(tmp)

        c21 = MaxPooling3D(pool_size=pool_size, name='pool_1')(end_11)
        c21 = Conv3D(16, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_conn_2.1', padding='same')(c21)

        end_21 = Add()([z2, c21])

        # Third block (down)
        tmp = Activation('relu')(end_21)
        tmp = Conv3D(32, (2, 2, 2), strides=(2, 2, 2), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='downpool_2', padding='same')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_3.1',
                     padding='same')(tmp)
        tmp = Activation('relu')(tmp)
        z3 = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_3.2',
                    padding='same')(tmp)

        c31 = MaxPooling3D(pool_size=pool_size, name='pool_2')(end_21)
        c31 = Conv3D(32, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_conn_3.1', padding='same')(c31)

        end_31 = Add()([z3, c31])

        # Fourth block (down)
        tmp = Activation('relu')(end_31)
        tmp = Conv3D(64, (2, 2, 2), strides=(2, 2, 2), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='downpool_3', padding='same')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_4.1',
                     padding='same')(tmp)
        tmp = Activation('relu')(tmp)
        z4 = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_4.2',
                    padding='same')(tmp)

        c41 = MaxPooling3D(pool_size=pool_size, name='pool_3')(end_31)
        c41 = Conv3D(64, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_conn_4.1', padding='same')(c41)

        end_41 = Add()([z4, c41])

        # Fifth block
        tmp = Activation('relu')(end_41)
        tmp = Conv3D(128, (2, 2, 2), strides=(2, 2, 2), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='downpool_4', padding='same')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(128, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_5.1',
                     padding='same')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(128, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_5.2',
                     padding='same')(tmp)  # inflection point

        c5 = MaxPooling3D(pool_size=pool_size, name='pool_4')(end_41)
        c5 = Conv3D(128, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_conn_5',
                    padding='same')(c5)

        end_5 = Add()([tmp, c5])

        # Fourth block (up)
        tmp = Activation('relu')(end_5)
        tmp = UpSampling3D(size=pool_size, name='up_4')(tmp)
        tmp = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_up_4',
                     padding='same')(tmp)
        tmp = Concatenate(axis=4)([tmp, z4])
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_4.3',
                     padding='same')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_4.4',
                     padding='same')(tmp)

        c42 = UpSampling3D(size=pool_size, name='up_4conn')(end_5)
        c42 = Conv3D(64, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_conn_4.2', padding='same')(c42)

        end_42 = Add()([tmp, c42])

        # Third block (up)
        tmp = Activation('relu')(end_42)
        tmp = UpSampling3D(size=pool_size, name='up_3')(tmp)
        tmp = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_up_3',
                     padding='same')(tmp)
        tmp = Concatenate(axis=4)([tmp, z3])
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_3.3',
                     padding='same')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_3.4',
                     padding='same')(tmp)

        c32 = UpSampling3D(size=pool_size, name='up_3conn')(end_42)
        c32 = Conv3D(32, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_conn_3.2', padding='same')(c32)

        end_32 = Add()([tmp, c32])

        # Second block (up)
        tmp = Activation('relu')(end_32)
        tmp = UpSampling3D(size=pool_size, name='up_2')(tmp)
        tmp = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_up_2',
                     padding='same')(tmp)
        tmp = Concatenate(axis=4)([tmp, z2])
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_2.3',
                     padding='same')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_2.4',
                     padding='same')(tmp)

        c22 = UpSampling3D(size=pool_size, name='up_2conn')(end_32)
        c22 = Conv3D(16, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_conn_2.2', padding='same')(c22)

        end_22 = Add()([tmp, c22])

        # First block (up)
        tmp = Activation('relu')(end_22)
        tmp = UpSampling3D(size=pool_size, name='up_1')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_up_1',
                     padding='same')(tmp)
        tmp = Concatenate(axis=4)([tmp, z1])
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.3',
                     padding='same')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.4',
                     padding='same')(tmp)

        c12 = UpSampling3D(size=pool_size, name='up_1_conn')(end_22)
        c12 = Conv3D(8, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_conn_1.2',
                     padding='same')(c12)

        end_12 = Add()([tmp, c12])

        # Final convolution
        tmp = Concatenate(axis=4)([end_12, end_11_d])
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_pre_softmax', padding='same')(tmp)
        in_softmax = Activation('relu')(tmp)


        classification = Conv3D(num_classes, (1, 1, 1), kernel_initializer=initializer,
                                name='final_convolution_1x1x1')(in_softmax)


        true_mask = Input(shape=segment_dimensions + (1,), name='V-net_mask1')

        mask1 = Lambda(repeat_channels(num_classes), output_shape=repeat_channels_shape(num_classes),
                       name='repeat_mask')(true_mask)


        cmp_mask = Lambda(complementary_mask, output_shape=segment_dimensions + (1,),
                          name='complementary_tumor_mask')(true_mask)
        cmp_mask = Concatenate()([cmp_mask,
                                  Lambda(lambda x: K.zeros_like(x))(cmp_mask),
                                  Lambda(lambda x: K.zeros_like(x))(cmp_mask),
                                  Lambda(lambda x: K.zeros_like(x))(cmp_mask)])

        y = softmax_activation(classification)
        y = Multiply(name='mask_background')([y, mask1])
        y = Add(name='label_background')([y, cmp_mask])

        model = Model(inputs=[x, true_mask], outputs=y)


        return model, output_shape


    @staticmethod
    def v_net_BN_patches(num_modalities, segment_dimensions, num_classes, shortcut_input=False,
              l1=0.0, l2=0.0):
        """
        U-Net based architecture for segmentation (http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)
            - Convolutional layers: 3x3x3 filters, stride 1, same border
            - Max pooling: 2x2x2
            - Upsampling layers: 2x2x2
            - Activations: ReLU
            - Classification layer: 1x1x1 kernel, and there are as many kernels as classes to be predicted. Element-wise
              softmax as activation.
            - Loss: 3D categorical cross-entropy
            - Optimizer: Adam with lr=0.001, beta_1=0.9, beta_2=0.999 and epsilon=10 ** (-4)
            - Regularization: L1=0.000001, L2=0.0001
            - Weights initialization: He et al normal initialization (https://arxiv.org/abs/1502.01852)

        Returns
        -------
        keras.model
            Compiled Sequential model
        tuple (dim1, dim2, dim3)
            Output shape computed from segment_dimensions and the convolutional architecture
        """
        if not isinstance(segment_dimensions, tuple) or len(segment_dimensions) != 3:
            raise ValueError('segment_dimensions must be a tuple with length 3, specifying the shape of the '
                             'input segment')

        # for dim in segment_dimensions:
        #     assert dim % 32 == 0  # As there are 5 (2, 2, 2) max-poolings, 2 ** 5 is the minimum input size

        # Hyperaparametre values

        initializer = 'he_normal'
        pool_size = (2, 2, 2)

        # Compute input shape, receptive field and output shape after softmax activation
        input_shape = segment_dimensions + (num_modalities,)
        output_shape = segment_dimensions

        # Activations, regularizers and optimizers
        softmax_activation = Lambda(elementwise_softmax_3d, name='Softmax')
        regularizer = l1_l2(l1=l1, l2=l2)

        # Architecture definition
        # INPUT
        x = Input(shape=input_shape, name='V-net_input')


        # First block (down)
        first_conv = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer,
                            name='conv_initial', padding='same')(x)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.1')(first_conv)
        tmp = Activation('relu')(tmp)
        z1 = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.1',
                    padding='same')(tmp)

        c11 = Conv3D(8, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_conn_1.1',
                     padding='same')(x)
        end_11 = Add()([z1, c11])






        # First_b block (down)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.2_b')(end_11)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.2_b',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.3_b')(tmp)
        tmp = Activation('relu')(tmp)
        z1_b = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.3_b',
                    padding='same')(tmp)


        end_11_b = Add()([z1_b, end_11])




        # First_d block (down)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.2_c')(end_11_b)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.2_c',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.3_c')(tmp)
        tmp = Activation('relu')(tmp)
        z1_c = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.3_c',
                    padding='same')(tmp)


        end_11_c = Add()([z1_c, end_11_b])



        # First_c block (down)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.2_d')(end_11_c)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.2_d',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.3_d')(tmp)
        tmp = Activation('relu')(tmp)
        z1_d = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.3_d',
                    padding='same')(tmp)

        end_11_d = Add()([z1_d, end_11_c])






        # Second block (down)
        tmp = BatchNormalization(axis=4, name='batch_norm_2.1')(end_11)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(16, (2, 2, 2), strides=(2, 2, 2), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='downpool_1', padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_2.2')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_2.1',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_2.3')(tmp)
        tmp = Activation('relu')(tmp)
        z2 = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_2.2',
                    padding='same')(tmp)

        c21 = MaxPooling3D(pool_size=pool_size, name='pool_1')(end_11)
        c21 = Conv3D(16, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_conn_2.1', padding='same')(c21)

        end_21 = Add()([z2, c21])

        # Third block (down)
        tmp = BatchNormalization(axis=4, name='batch_norm_3.1')(end_21)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(32, (2, 2, 2), strides=(2, 2, 2), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='downpool_2', padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_3.2')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_3.1',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_3.3')(tmp)
        tmp = Activation('relu')(tmp)
        z3 = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_3.2',
                    padding='same')(tmp)

        c31 = MaxPooling3D(pool_size=pool_size, name='pool_2')(end_21)
        c31 = Conv3D(32, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_conn_3.1', padding='same')(c31)

        end_31 = Add()([z3, c31])

        # Fourth block (down)
        tmp = BatchNormalization(axis=4, name='batch_norm_4.1')(end_31)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(64, (2, 2, 2), strides=(2, 2, 2), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='downpool_3', padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_4.2')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_4.1',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_4.3')(tmp)
        tmp = Activation('relu')(tmp)
        z4 = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_4.2',
                    padding='same')(tmp)

        c41 = MaxPooling3D(pool_size=pool_size, name='pool_3')(end_31)
        c41 = Conv3D(64, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_conn_4.1', padding='same')(c41)

        end_41 = Add()([z4, c41])

        # Fifth block
        tmp = BatchNormalization(axis=4, name='batch_norm_5.1')(end_41)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(128, (2, 2, 2), strides=(2, 2, 2), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='downpool_4', padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_5.2')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(128, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_5.1',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_5.3')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(128, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_5.2',
                     padding='same')(tmp)  # inflection point

        c5 = MaxPooling3D(pool_size=pool_size, name='pool_4')(end_41)
        c5 = Conv3D(128, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_conn_5',
                    padding='same')(c5)

        end_5 = Add()([tmp, c5])

        # Fourth block (up)
        tmp = BatchNormalization(axis=4, name='batch_norm_4.4')(end_5)
        tmp = Activation('relu')(tmp)
        tmp = UpSampling3D(size=pool_size, name='up_4')(tmp)
        tmp = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_up_4',
                     padding='same')(tmp)
        tmp = Concatenate(axis=4)([tmp, z4])
        tmp = BatchNormalization(axis=4, name='batch_norm_4.5')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_4.3',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_4.6')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_4.4',
                     padding='same')(tmp)

        c42 = UpSampling3D(size=pool_size, name='up_4conn')(end_5)
        c42 = Conv3D(64, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_conn_4.2', padding='same')(c42)

        end_42 = Add()([tmp, c42])

        # Third block (up)
        tmp = BatchNormalization(axis=4, name='batch_norm_3.4')(end_42)
        tmp = Activation('relu')(tmp)
        tmp = UpSampling3D(size=pool_size, name='up_3')(tmp)
        tmp = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_up_3',
                     padding='same')(tmp)
        tmp = Concatenate(axis=4)([tmp, z3])
        tmp = BatchNormalization(axis=4, name='batch_norm_3.5')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_3.3',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_3.6')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_3.4',
                     padding='same')(tmp)

        c32 = UpSampling3D(size=pool_size, name='up_3conn')(end_42)
        c32 = Conv3D(32, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_conn_3.2', padding='same')(c32)

        end_32 = Add()([tmp, c32])

        # Second block (up)
        tmp = BatchNormalization(axis=4, name='batch_norm_2.4')(end_32)
        tmp = Activation('relu')(tmp)
        tmp = UpSampling3D(size=pool_size, name='up_2')(tmp)
        tmp = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_up_2',
                     padding='same')(tmp)
        tmp = Concatenate(axis=4)([tmp, z2])
        tmp = BatchNormalization(axis=4, name='batch_norm_2.5')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_2.3',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_2.6')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_2.4',
                     padding='same')(tmp)

        c22 = UpSampling3D(size=pool_size, name='up_2conn')(end_32)
        c22 = Conv3D(16, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_conn_2.2', padding='same')(c22)

        end_22 = Add()([tmp, c22])

        # First block (up)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.4')(end_22)
        tmp = Activation('relu')(tmp)
        tmp = UpSampling3D(size=pool_size, name='up_1')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_up_1',
                     padding='same')(tmp)
        tmp = Concatenate(axis=4)([tmp, z1])
        tmp = BatchNormalization(axis=4, name='batch_norm_1.5')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.3',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.6')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.4',
                     padding='same')(tmp)

        c12 = UpSampling3D(size=pool_size, name='up_1_conn')(end_22)
        c12 = Conv3D(8, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_conn_1.2',
                     padding='same')(c12)

        end_12 = Add()([tmp, c12])

        # Final convolution

        tmp = BatchNormalization(axis=4, name='batch_norm_1.7')(Concatenate(axis=4)([end_12, end_11_d]))
        if shortcut_input:
            tmp = Concatenate(axis=4)([tmp, Lambda(lambda x: x[:, :, :, :, 0:1])(x)])

        tmp = Activation('relu')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_pre_softmax', padding='same')(tmp)

        tmp = BatchNormalization(axis=4, name='batch_norm_pre_softmax')(tmp)

        in_softmax = Activation('relu')(tmp)

        classification = Conv3D(num_classes, (1, 1, 1), kernel_initializer=initializer,
                                name='final_convolution_1x1x1')(in_softmax)

        y = softmax_activation(classification)

        model = Model(inputs=[x], outputs=y)

        return model, output_shape

    @staticmethod
    def v_net_BN_patches_sr2(num_modalities, segment_dimensions, num_classes, shortcut_input=False,
                            l1=0.0, l2=0.0, mode='train'):
        """
        U-Net based architecture for segmentation (http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)
            - Convolutional layers: 3x3x3 filters, stride 1, same border
            - Max pooling: 2x2x2
            - Upsampling layers: 2x2x2
            - Activations: ReLU
            - Classification layer: 1x1x1 kernel, and there are as many kernels as classes to be predicted. Element-wise
              softmax as activation.
            - Loss: 3D categorical cross-entropy
            - Optimizer: Adam with lr=0.001, beta_1=0.9, beta_2=0.999 and epsilon=10 ** (-4)
            - Regularization: L1=0.000001, L2=0.0001
            - Weights initialization: He et al normal initialization (https://arxiv.org/abs/1502.01852)

        Returns
        -------
        keras.model
            Compiled Sequential model
        tuple (dim1, dim2, dim3)
            Output shape computed from segment_dimensions and the convolutional architecture
        """
        print(shortcut_input)
        if not isinstance(segment_dimensions, tuple) or len(segment_dimensions) != 3:
            raise ValueError('segment_dimensions must be a tuple with length 3, specifying the shape of the '
                             'input segment')

        # for dim in segment_dimensions:
        #     assert dim % 32 == 0  # As there are 5 (2, 2, 2) max-poolings, 2 ** 5 is the minimum input size

        # Hyperaparametre values

        initializer = 'he_normal'
        pool_size = (2, 2, 2)

        # Compute input shape, receptive field and output shape after softmax activation
        input_shape = segment_dimensions + (num_modalities,)
        output_shape = segment_dimensions

        # Activations, regularizers and optimizers
        softmax_activation = Lambda(elementwise_softmax_3d, name='Softmax')
        regularizer = l1_l2(l1=l1, l2=l2)

        # Architecture definition
        # INPUT
        x = Input(shape=input_shape, name='V-net_input')

        # First block (down)
        first_conv = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer,
                            name='conv_initial', padding='same')(x)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.1')(first_conv)
        tmp = Activation('relu')(tmp)
        z1 = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.1',
                    padding='same')(tmp)

        c11 = Conv3D(8, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_conn_1.1',
                     padding='same')(x)
        end_11 = Add()([z1, c11])

        # First_b block (up)
        x_up = UpSampling3D(size=pool_size, name='sr_up')(x)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.2_b')(x_up)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(4, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.2_b',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.3_b')(tmp)
        tmp = Activation('relu')(tmp)
        z1_b = Conv3D(4, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.3_b',
                      padding='same')(tmp)

        x_up = Conv3D(4, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                      name='conv_conn_0.1', padding='same')(x_up)
        end_11_b = Add()([z1_b, x_up])

        # First_c block (up)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.2_c')(end_11_b)
        tmp = Activation('relu')(tmp)
        z1_c = Conv3D(4, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.2_c',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.3_c')(tmp)
        tmp = Activation('relu')(tmp)
        z1_c = Conv3D(4, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.3_c',
                      padding='same')(tmp)

        end_11_c = Add()([z1_c, end_11_b])

        # First_d block (up)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.2_d')(end_11_c)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(4, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.2_d',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.3_d')(tmp)
        tmp = Activation('relu')(tmp)
        z1_d = Conv3D(4, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.3_d',
                      padding='same')(tmp)

        # end_11_c = Conv3D(4, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
        #               name='conv_conn_1.4_d', padding='same')(end_11_c)
        # end_11_c = MaxPooling3D(pool_size = pool_size)(end_11_c)
        end_11_d = Add()([z1_d, end_11_c])



        # Second block (down)
        tmp = BatchNormalization(axis=4, name='batch_norm_2.1')(end_11)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(16, (2, 2, 2), strides=(2, 2, 2), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='downpool_1', padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_2.2')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_2.1',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_2.3')(tmp)
        tmp = Activation('relu')(tmp)
        z2 = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_2.2',
                    padding='same')(tmp)

        c21 = MaxPooling3D(pool_size=pool_size, name='pool_1')(end_11)
        c21 = Conv3D(16, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_conn_2.1', padding='same')(c21)

        end_21 = Add()([z2, c21])

        # Third block (down)
        tmp = BatchNormalization(axis=4, name='batch_norm_3.1')(end_21)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(32, (2, 2, 2), strides=(2, 2, 2), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='downpool_2', padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_3.2')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_3.1',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_3.3')(tmp)
        tmp = Activation('relu')(tmp)
        z3 = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_3.2',
                    padding='same')(tmp)

        c31 = MaxPooling3D(pool_size=pool_size, name='pool_2')(end_21)
        c31 = Conv3D(32, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_conn_3.1', padding='same')(c31)

        end_31 = Add()([z3, c31])

        # Fourth block (down)
        tmp = BatchNormalization(axis=4, name='batch_norm_4.1')(end_31)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(64, (2, 2, 2), strides=(2, 2, 2), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='downpool_3', padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_4.2')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_4.1',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_4.3')(tmp)
        tmp = Activation('relu')(tmp)
        z4 = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_4.2',
                    padding='same')(tmp)

        c41 = MaxPooling3D(pool_size=pool_size, name='pool_3')(end_31)
        c41 = Conv3D(64, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_conn_4.1', padding='same')(c41)

        end_41 = Add()([z4, c41])

        # Fifth block
        tmp = BatchNormalization(axis=4, name='batch_norm_5.1')(end_41)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(128, (2, 2, 2), strides=(2, 2, 2), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='downpool_4', padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_5.2')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(128, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_5.1',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_5.3')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(128, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_5.2',
                     padding='same')(tmp)  # inflection point

        c5 = MaxPooling3D(pool_size=pool_size, name='pool_4')(end_41)
        c5 = Conv3D(128, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_conn_5',
                    padding='same')(c5)

        end_5 = Add()([tmp, c5])

        # Fourth block (up)
        tmp = BatchNormalization(axis=4, name='batch_norm_4.4')(end_5)
        tmp = Activation('relu')(tmp)
        tmp = UpSampling3D(size=pool_size, name='up_4')(tmp)
        tmp = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_up_4',
                     padding='same')(tmp)
        tmp = Concatenate(axis=4)([tmp, z4])
        tmp = BatchNormalization(axis=4, name='batch_norm_4.5')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_4.3',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_4.6')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_4.4',
                     padding='same')(tmp)

        c42 = UpSampling3D(size=pool_size, name='up_4conn')(end_5)
        c42 = Conv3D(64, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_conn_4.2', padding='same')(c42)

        end_42 = Add()([tmp, c42])

        # Third block (up)
        tmp = BatchNormalization(axis=4, name='batch_norm_3.4')(end_42)
        tmp = Activation('relu')(tmp)
        tmp = UpSampling3D(size=pool_size, name='up_3')(tmp)
        tmp = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_up_3',
                     padding='same')(tmp)
        tmp = Concatenate(axis=4)([tmp, z3])
        tmp = BatchNormalization(axis=4, name='batch_norm_3.5')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_3.3',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_3.6')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_3.4',
                     padding='same')(tmp)

        c32 = UpSampling3D(size=pool_size, name='up_3conn')(end_42)
        c32 = Conv3D(32, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_conn_3.2', padding='same')(c32)

        end_32 = Add()([tmp, c32])

        # Second block (up)
        tmp = BatchNormalization(axis=4, name='batch_norm_2.4')(end_32)
        tmp = Activation('relu')(tmp)
        tmp = UpSampling3D(size=pool_size, name='up_2')(tmp)
        tmp = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_up_2',
                     padding='same')(tmp)
        tmp = Concatenate(axis=4)([tmp, z2])
        tmp = BatchNormalization(axis=4, name='batch_norm_2.5')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_2.3',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_2.6')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_2.4',
                     padding='same')(tmp)

        c22 = UpSampling3D(size=pool_size, name='up_2conn')(end_32)
        c22 = Conv3D(16, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_conn_2.2', padding='same')(c22)

        end_22 = Add()([tmp, c22])

        # First block (up)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.4')(end_22)
        tmp = Activation('relu')(tmp)
        tmp = UpSampling3D(size=pool_size, name='up_1')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_up_1',
                     padding='same')(tmp)
        tmp = Concatenate(axis=4)([tmp, z1])
        tmp = BatchNormalization(axis=4, name='batch_norm_1.5')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.3',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.6')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.4',
                     padding='same')(tmp)

        c12 = UpSampling3D(size=pool_size, name='up_1_conn')(end_22)
        c12 = Conv3D(8, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_conn_1.2',
                     padding='same')(c12)

        end_12 = Add()([tmp, c12])

        # Final convolution
        tmp = MaxPooling3D(pool_size=pool_size, name='sr_down')(end_11_d)
        tmp = Concatenate(axis=4)([end_12, tmp])

        if shortcut_input:
            tmp = Concatenate(axis=4)([tmp, x])
        tmp = BatchNormalization(axis=4, name='batch_norm_1.7')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_pre_softmax', padding='same')(tmp)

        tmp = BatchNormalization(axis=4, name='batch_norm_pre_softmax')(tmp)

        in_softmax = Activation('relu')(tmp)

        classification = Conv3D(num_classes, (1, 1, 1), kernel_initializer=initializer,
                                name='final_convolution_1x1x1')(in_softmax)

        y = softmax_activation(classification)

        if mode == 'train':
            model = Model(inputs=[x], outputs=y)
        else:
            true_mask = Input(shape=segment_dimensions + (1,), name='V-net_mask1')
            mask1 = Lambda(repeat_channels(num_classes), output_shape=repeat_channels_shape(num_classes),
                           name='repeat_mask')(true_mask)
            cmp_mask = Lambda(complementary_mask, output_shape=segment_dimensions + (1,),
                              name='complementary_tumor_mask')(true_mask)
            cmp_mask = Concatenate()([cmp_mask,
                                      Lambda(lambda x: K.zeros_like(x))(cmp_mask),
                                      Lambda(lambda x: K.zeros_like(x))(cmp_mask),
                                      Lambda(lambda x: K.zeros_like(x))(cmp_mask)
                                      ])

            y = Multiply(name='mask_background')([y, mask1])
            y = Add(name='label_background')([y, cmp_mask])

            model = Model(inputs=[x, true_mask], outputs=y)

        return model, output_shape

    @staticmethod
    def v_net_BN_patches_sr(num_modalities, segment_dimensions, num_classes, shortcut_input=False,
                            l1=0.0, l2=0.0, mode='train'):
        """
        U-Net based architecture for segmentation (http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)
            - Convolutional layers: 3x3x3 filters, stride 1, same border
            - Max pooling: 2x2x2
            - Upsampling layers: 2x2x2
            - Activations: ReLU
            - Classification layer: 1x1x1 kernel, and there are as many kernels as classes to be predicted. Element-wise
              softmax as activation.
            - Loss: 3D categorical cross-entropy
            - Optimizer: Adam with lr=0.001, beta_1=0.9, beta_2=0.999 and epsilon=10 ** (-4)
            - Regularization: L1=0.000001, L2=0.0001
            - Weights initialization: He et al normal initialization (https://arxiv.org/abs/1502.01852)

        Returns
        -------
        keras.model
            Compiled Sequential model
        tuple (dim1, dim2, dim3)
            Output shape computed from segment_dimensions and the convolutional architecture
        """
        print(shortcut_input)
        if not isinstance(segment_dimensions, tuple) or len(segment_dimensions) != 3:
            raise ValueError('segment_dimensions must be a tuple with length 3, specifying the shape of the '
                             'input segment')

        # for dim in segment_dimensions:
        #     assert dim % 32 == 0  # As there are 5 (2, 2, 2) max-poolings, 2 ** 5 is the minimum input size

        # Hyperaparametre values

        initializer = 'he_normal'
        pool_size = (2, 2, 2)

        # Compute input shape, receptive field and output shape after softmax activation
        input_shape = segment_dimensions + (num_modalities,)
        output_shape = segment_dimensions

        # Activations, regularizers and optimizers
        softmax_activation = Lambda(elementwise_softmax_3d, name='Softmax')
        regularizer = l1_l2(l1=l1, l2=l2)

        # Architecture definition
        # INPUT
        x = Input(shape=input_shape, name='V-net_input')

        # First block (down)
        first_conv = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer,
                            name='conv_initial', padding='same')(x)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.1')(first_conv)
        tmp = Activation('relu')(tmp)
        z1 = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.1',
                    padding='same')(tmp)

        c11 = Conv3D(8, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_conn_1.1',
                     padding='same')(x)
        end_11 = Add()([z1, c11])

        # First_b block (down)
        x_up = UpSampling3D(size=pool_size, name='sr_up')(x)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.2_b')(x_up)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(4, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.2_b',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.3_b')(tmp)
        tmp = Activation('relu')(tmp)
        z1_b = Conv3D(4, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.3_b',
                      padding='same')(tmp)

        x_up = Conv3D(4, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                      name='conv_conn_0.1', padding='same')(x_up)
        end_11_b = Add()([z1_b, x_up])

        # First_c block (down)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.2_c')(end_11_b)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(4, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.2_c',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.3_c')(tmp)
        tmp = Activation('relu')(tmp)
        z1_c = Conv3D(4, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.3_c',
                      padding='same')(tmp)

        end_11_c = Add()([z1_c, end_11_b])

        # First_d block (down)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.2_d')(end_11_c)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(4, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.2_d',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.3_d')(tmp)
        tmp = Activation('relu')(tmp)
        z1_d = Conv3D(4, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.3_d',
                      padding='same')(tmp)

        end_11_d = Add()([z1_d, end_11_c])

        # Second block (down)
        tmp = BatchNormalization(axis=4, name='batch_norm_2.1')(end_11)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(16, (2, 2, 2), strides=(2, 2, 2), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='downpool_1', padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_2.2')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_2.1',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_2.3')(tmp)
        tmp = Activation('relu')(tmp)
        z2 = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_2.2',
                    padding='same')(tmp)

        c21 = MaxPooling3D(pool_size=pool_size, name='pool_1')(end_11)
        c21 = Conv3D(16, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_conn_2.1', padding='same')(c21)

        end_21 = Add()([z2, c21])

        # Third block (down)
        tmp = BatchNormalization(axis=4, name='batch_norm_3.1')(end_21)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(32, (2, 2, 2), strides=(2, 2, 2), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='downpool_2', padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_3.2')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_3.1',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_3.3')(tmp)
        tmp = Activation('relu')(tmp)
        z3 = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_3.2',
                    padding='same')(tmp)

        c31 = MaxPooling3D(pool_size=pool_size, name='pool_2')(end_21)
        c31 = Conv3D(32, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_conn_3.1', padding='same')(c31)

        end_31 = Add()([z3, c31])

        # Fourth block (down)
        tmp = BatchNormalization(axis=4, name='batch_norm_4.1')(end_31)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(64, (2, 2, 2), strides=(2, 2, 2), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='downpool_3', padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_4.2')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_4.1',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_4.3')(tmp)
        tmp = Activation('relu')(tmp)
        z4 = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_4.2',
                    padding='same')(tmp)

        c41 = MaxPooling3D(pool_size=pool_size, name='pool_3')(end_31)
        c41 = Conv3D(64, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_conn_4.1', padding='same')(c41)

        end_41 = Add()([z4, c41])

        # Fifth block
        tmp = BatchNormalization(axis=4, name='batch_norm_5.1')(end_41)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(128, (2, 2, 2), strides=(2, 2, 2), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='downpool_4', padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_5.2')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(128, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_5.1',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_5.3')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(128, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_5.2',
                     padding='same')(tmp)  # inflection point

        c5 = MaxPooling3D(pool_size=pool_size, name='pool_4')(end_41)
        c5 = Conv3D(128, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_conn_5',
                    padding='same')(c5)

        end_5 = Add()([tmp, c5])

        # Fourth block (up)
        tmp = BatchNormalization(axis=4, name='batch_norm_4.4')(end_5)
        tmp = Activation('relu')(tmp)
        tmp = UpSampling3D(size=pool_size, name='up_4')(tmp)
        tmp = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_up_4',
                     padding='same')(tmp)
        tmp = Concatenate(axis=4)([tmp, z4])
        tmp = BatchNormalization(axis=4, name='batch_norm_4.5')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_4.3',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_4.6')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_4.4',
                     padding='same')(tmp)

        c42 = UpSampling3D(size=pool_size, name='up_4conn')(end_5)
        c42 = Conv3D(64, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_conn_4.2', padding='same')(c42)

        end_42 = Add()([tmp, c42])

        # Third block (up)
        tmp = BatchNormalization(axis=4, name='batch_norm_3.4')(end_42)
        tmp = Activation('relu')(tmp)
        tmp = UpSampling3D(size=pool_size, name='up_3')(tmp)
        tmp = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_up_3',
                     padding='same')(tmp)
        tmp = Concatenate(axis=4)([tmp, z3])
        tmp = BatchNormalization(axis=4, name='batch_norm_3.5')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_3.3',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_3.6')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_3.4',
                     padding='same')(tmp)

        c32 = UpSampling3D(size=pool_size, name='up_3conn')(end_42)
        c32 = Conv3D(32, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_conn_3.2', padding='same')(c32)

        end_32 = Add()([tmp, c32])

        # Second block (up)
        tmp = BatchNormalization(axis=4, name='batch_norm_2.4')(end_32)
        tmp = Activation('relu')(tmp)
        tmp = UpSampling3D(size=pool_size, name='up_2')(tmp)
        tmp = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_up_2',
                     padding='same')(tmp)
        tmp = Concatenate(axis=4)([tmp, z2])
        tmp = BatchNormalization(axis=4, name='batch_norm_2.5')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_2.3',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_2.6')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_2.4',
                     padding='same')(tmp)

        c22 = UpSampling3D(size=pool_size, name='up_2conn')(end_32)
        c22 = Conv3D(16, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_conn_2.2', padding='same')(c22)

        end_22 = Add()([tmp, c22])

        # First block (up)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.4')(end_22)
        tmp = Activation('relu')(tmp)
        tmp = UpSampling3D(size=pool_size, name='up_1')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_up_1',
                     padding='same')(tmp)
        tmp = Concatenate(axis=4)([tmp, z1])
        tmp = BatchNormalization(axis=4, name='batch_norm_1.5')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.3',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.6')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.4',
                     padding='same')(tmp)

        c12 = UpSampling3D(size=pool_size, name='up_1_conn')(end_22)
        c12 = Conv3D(8, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_conn_1.2',
                     padding='same')(c12)

        end_12 = Add()([tmp, c12])

        # Final convolution
        tmp = MaxPooling3D(pool_size=pool_size, name='sr_down')(end_11_d)
        tmp = Concatenate(axis=4)([end_12, tmp])

        if shortcut_input:
            tmp = Concatenate(axis=4)([tmp, x])
        tmp = BatchNormalization(axis=4, name='batch_norm_1.7')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_pre_softmax', padding='same')(tmp)

        tmp = BatchNormalization(axis=4, name='batch_norm_pre_softmax')(tmp)

        in_softmax = Activation('relu')(tmp)

        classification = Conv3D(num_classes, (1, 1, 1), kernel_initializer=initializer,
                                name='final_convolution_1x1x1')(in_softmax)

        y = softmax_activation(classification)

        if mode == 'train':
            model = Model(inputs=[x], outputs=y)
        else:
            true_mask = Input(shape=segment_dimensions + (1,), name='V-net_mask1')
            mask1 = Lambda(repeat_channels(num_classes), output_shape=repeat_channels_shape(num_classes),
                           name='repeat_mask')(true_mask)
            cmp_mask = Lambda(complementary_mask, output_shape=segment_dimensions + (1,),
                              name='complementary_tumor_mask')(true_mask)
            cmp_mask = Concatenate()([cmp_mask,
                                      Lambda(lambda x: K.zeros_like(x))(cmp_mask),
                                      Lambda(lambda x: K.zeros_like(x))(cmp_mask),
                                      Lambda(lambda x: K.zeros_like(x))(cmp_mask)
                                      ])

            y = Multiply(name='mask_background')([y, mask1])
            y = Add(name='label_background')([y, cmp_mask])

            model = Model(inputs=[x, true_mask], outputs=y)

        return model, output_shape


    @staticmethod
    def v_net_BN_patches_sr_old(num_modalities, segment_dimensions, num_classes, shortcut_input=False,
                                l1=0.0, l2=0.0, mode='train'):
        """
        U-Net based architecture for segmentation (http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)
            - Convolutional layers: 3x3x3 filters, stride 1, same border
            - Max pooling: 2x2x2
            - Upsampling layers: 2x2x2
            - Activations: ReLU
            - Classification layer: 1x1x1 kernel, and there are as many kernels as classes to be predicted. Element-wise
              softmax as activation.
            - Loss: 3D categorical cross-entropy
            - Optimizer: Adam with lr=0.001, beta_1=0.9, beta_2=0.999 and epsilon=10 ** (-4)
            - Regularization: L1=0.000001, L2=0.0001
            - Weights initialization: He et al normal initialization (https://arxiv.org/abs/1502.01852)

        Returns
        -------
        keras.model
            Compiled Sequential model
        tuple (dim1, dim2, dim3)
            Output shape computed from segment_dimensions and the convolutional architecture
        """

        if not isinstance(segment_dimensions, tuple) or len(segment_dimensions) != 3:
            raise ValueError('segment_dimensions must be a tuple with length 3, specifying the shape of the '
                             'input segment')

        # for dim in segment_dimensions:
        #     assert dim % 32 == 0  # As there are 5 (2, 2, 2) max-poolings, 2 ** 5 is the minimum input size

        # Hyperaparametre values

        initializer = 'he_normal'
        pool_size = (2, 2, 2)

        # Compute input shape, receptive field and output shape after softmax activation
        input_shape = segment_dimensions + (num_modalities,)
        output_shape = segment_dimensions

        # Activations, regularizers and optimizers
        softmax_activation = Lambda(elementwise_softmax_3d, name='Softmax')
        regularizer = l1_l2(l1=l1, l2=l2)

        # Architecture definition
        # INPUT
        x = Input(shape=input_shape, name='V-net_input')

        # First block (down)
        first_conv = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer,
                            name='conv_initial', padding='same')(x)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.1')(first_conv)
        tmp = Activation('relu')(tmp)
        z1 = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.1',
                    padding='same')(tmp)

        c11 = Conv3D(8, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_conn_1.1',
                     padding='same')(x)
        end_11 = Add()([z1, c11])

        # First_b block (down)
        x_up = UpSampling3D(size=pool_size, name='sr_up')(x)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.2_b')(x_up)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.2_b',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.3_b')(tmp)
        tmp = Activation('relu')(tmp)
        z1_b = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.3_b',
                      padding='same')(tmp)

        x_up = Conv3D(8, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                      name='conv_conn_0.1', padding='same')(x_up)
        end_11_b = Add()([z1_b, x_up])

        # # First_c block (down)
        # tmp = BatchNormalization(axis=4, name='batch_norm_1.2_c')(end_11_b)
        # tmp = Activation('relu')(tmp)
        # tmp = Conv3D(4, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.2_c',
        #              padding='same')(tmp)
        # tmp = BatchNormalization(axis=4, name='batch_norm_1.3_c')(tmp)
        # tmp = Activation('relu')(tmp)
        # z1_c = Conv3D(4, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.3_c',
        #               padding='same')(tmp)
        #
        # end_11_c = Add()([z1_c, end_11_b])
        #
        # # First_d block (down)
        # tmp = BatchNormalization(axis=4, name='batch_norm_1.2_d')(end_11_c)
        # tmp = Activation('relu')(tmp)
        # tmp = Conv3D(4, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.2_d',
        #              padding='same')(tmp)
        # tmp = BatchNormalization(axis=4, name='batch_norm_1.3_d')(tmp)
        # tmp = Activation('relu')(tmp)
        # z1_d = Conv3D(4, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.3_d',
        #               padding='same')(tmp)

        # end_11_d = Add()([z1_d, end_11_c])

        # Second block (down)
        tmp = BatchNormalization(axis=4, name='batch_norm_2.1')(end_11)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(16, (2, 2, 2), strides=(2, 2, 2), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='downpool_1', padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_2.2')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_2.1',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_2.3')(tmp)
        tmp = Activation('relu')(tmp)
        z2 = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_2.2',
                    padding='same')(tmp)

        c21 = MaxPooling3D(pool_size=pool_size, name='pool_1')(end_11)
        c21 = Conv3D(16, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_conn_2.1', padding='same')(c21)

        end_21 = Add()([z2, c21])

        # Third block (down)
        tmp = BatchNormalization(axis=4, name='batch_norm_3.1')(end_21)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(32, (2, 2, 2), strides=(2, 2, 2), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='downpool_2', padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_3.2')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_3.1',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_3.3')(tmp)
        tmp = Activation('relu')(tmp)
        z3 = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_3.2',
                    padding='same')(tmp)

        c31 = MaxPooling3D(pool_size=pool_size, name='pool_2')(end_21)
        c31 = Conv3D(32, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_conn_3.1', padding='same')(c31)

        end_31 = Add()([z3, c31])

        # Fourth block (down)
        tmp = BatchNormalization(axis=4, name='batch_norm_4.1')(end_31)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(64, (2, 2, 2), strides=(2, 2, 2), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='downpool_3', padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_4.2')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_4.1',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_4.3')(tmp)
        tmp = Activation('relu')(tmp)
        z4 = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_4.2',
                    padding='same')(tmp)

        c41 = MaxPooling3D(pool_size=pool_size, name='pool_3')(end_31)
        c41 = Conv3D(64, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_conn_4.1', padding='same')(c41)

        end_41 = Add()([z4, c41])

        # Fifth block
        tmp = BatchNormalization(axis=4, name='batch_norm_5.1')(end_41)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(128, (2, 2, 2), strides=(2, 2, 2), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='downpool_4', padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_5.2')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(128, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_5.1',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_5.3')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(128, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_5.2',
                     padding='same')(tmp)  # inflection point

        c5 = MaxPooling3D(pool_size=pool_size, name='pool_4')(end_41)
        c5 = Conv3D(128, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_conn_5',
                    padding='same')(c5)

        end_5 = Add()([tmp, c5])

        # Fourth block (up)
        tmp = BatchNormalization(axis=4, name='batch_norm_4.4')(end_5)
        tmp = Activation('relu')(tmp)
        tmp = UpSampling3D(size=pool_size, name='up_4')(tmp)
        tmp = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_up_4',
                     padding='same')(tmp)
        tmp = Concatenate(axis=4)([tmp, z4])
        tmp = BatchNormalization(axis=4, name='batch_norm_4.5')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_4.3',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_4.6')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_4.4',
                     padding='same')(tmp)

        c42 = UpSampling3D(size=pool_size, name='up_4conn')(end_5)
        c42 = Conv3D(64, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_conn_4.2', padding='same')(c42)

        end_42 = Add()([tmp, c42])

        # Third block (up)
        tmp = BatchNormalization(axis=4, name='batch_norm_3.4')(end_42)
        tmp = Activation('relu')(tmp)
        tmp = UpSampling3D(size=pool_size, name='up_3')(tmp)
        tmp = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_up_3',
                     padding='same')(tmp)
        tmp = Concatenate(axis=4)([tmp, z3])
        tmp = BatchNormalization(axis=4, name='batch_norm_3.5')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_3.3',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_3.6')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_3.4',
                     padding='same')(tmp)

        c32 = UpSampling3D(size=pool_size, name='up_3conn')(end_42)
        c32 = Conv3D(32, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_conn_3.2', padding='same')(c32)

        end_32 = Add()([tmp, c32])

        # Second block (up)
        tmp = BatchNormalization(axis=4, name='batch_norm_2.4')(end_32)
        tmp = Activation('relu')(tmp)
        tmp = UpSampling3D(size=pool_size, name='up_2')(tmp)
        tmp = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_up_2',
                     padding='same')(tmp)
        tmp = Concatenate(axis=4)([tmp, z2])
        tmp = BatchNormalization(axis=4, name='batch_norm_2.5')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_2.3',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_2.6')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_2.4',
                     padding='same')(tmp)

        c22 = UpSampling3D(size=pool_size, name='up_2conn')(end_32)
        c22 = Conv3D(16, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_conn_2.2', padding='same')(c22)

        end_22 = Add()([tmp, c22])

        # First block (up)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.4')(end_22)
        tmp = Activation('relu')(tmp)
        tmp = UpSampling3D(size=pool_size, name='up_1')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_up_1',
                     padding='same')(tmp)
        tmp = Concatenate(axis=4)([tmp, z1])
        tmp = BatchNormalization(axis=4, name='batch_norm_1.5')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.3',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.6')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.4',
                     padding='same')(tmp)

        c12 = UpSampling3D(size=pool_size, name='up_1_conn')(end_22)
        c12 = Conv3D(8, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_conn_1.2',
                     padding='same')(c12)

        end_12 = Add()([tmp, c12])

        # Final convolution
        tmp = MaxPooling3D(pool_size=pool_size, name='sr_down')(end_11_b)
        tmp = Concatenate(axis=4)([end_12, tmp])
        tmp = BatchNormalization(axis=4, name='batch_norm_1.7')(tmp)
        if shortcut_input:
            tmp = Concatenate(axis=4)([tmp, Lambda(lambda x: x[:, :, :, :, 0:1])(x)])

        tmp = Activation('relu')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_pre_softmax', padding='same')(tmp)

        tmp = BatchNormalization(axis=4, name='batch_norm_pre_softmax')(tmp)

        in_softmax = Activation('relu')(tmp)

        classification = Conv3D(num_classes, (1, 1, 1), kernel_initializer=initializer,
                                name='final_convolution_1x1x1')(in_softmax)

        y = softmax_activation(classification)

        if mode == 'train':
            model = Model(inputs=[x], outputs=y)
        else:
            true_mask = Input(shape=segment_dimensions + (1,), name='V-net_mask1')
            mask1 = Lambda(repeat_channels(num_classes), output_shape=repeat_channels_shape(num_classes),
                           name='repeat_mask')(true_mask)
            cmp_mask = Lambda(complementary_mask, output_shape=segment_dimensions + (1,),
                              name='complementary_tumor_mask')(true_mask)
            cmp_mask = Concatenate()([cmp_mask,
                                      Lambda(lambda x: K.zeros_like(x))(cmp_mask),
                                      Lambda(lambda x: K.zeros_like(x))(cmp_mask),
                                      Lambda(lambda x: K.zeros_like(x))(cmp_mask)
                                      ])

            y = Multiply(name='mask_background')([y, mask1])
            y = Add(name='label_background')([y, cmp_mask])

            model = Model(inputs=[x, true_mask], outputs=y)

        return model, output_shape


    @staticmethod
    def single_path(num_modalities, segment_dimensions, num_classes, shortcut_input=False,
                 l1=0.0, l2=0.0):
        """
        U-Net based architecture for segmentation (http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)
            - Convolutional layers: 3x3x3 filters, stride 1, same border
            - Max pooling: 2x2x2
            - Upsampling layers: 2x2x2
            - Activations: ReLU
            - Classification layer: 1x1x1 kernel, and there are as many kernels as classes to be predicted. Element-wise
              softmax as activation.
            - Loss: 3D categorical cross-entropy
            - Optimizer: Adam with lr=0.001, beta_1=0.9, beta_2=0.999 and epsilon=10 ** (-4)
            - Regularization: L1=0.000001, L2=0.0001
            - Weights initialization: He et al normal initialization (https://arxiv.org/abs/1502.01852)

        Returns
        -------
        keras.model
            Compiled Sequential model
        tuple (dim1, dim2, dim3)
            Output shape computed from segment_dimensions and the convolutional architecture
        """
        if not isinstance(segment_dimensions, tuple) or len(segment_dimensions) != 3:
            raise ValueError('segment_dimensions must be a tuple with length 3, specifying the shape of the '
                             'input segment')

        # for dim in segment_dimensions:
        #     assert dim % 32 == 0  # As there are 5 (2, 2, 2) max-poolings, 2 ** 5 is the minimum input size

        # Hyperaparametre values

        initializer = 'he_normal'


        # Compute input shape, receptive field and output shape after softmax activation
        input_shape = segment_dimensions + (num_modalities,)
        output_shape = segment_dimensions

        # Activations, regularizers and optimizers
        softmax_activation = Lambda(elementwise_softmax_3d, name='Softmax')
        regularizer = l1_l2(l1=l1, l2=l2)

        # Architecture definition
        # INPUT
        x = Input(shape=input_shape, name='V-net_input')

        # First block (down)
        first_conv = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer,
                            name='conv_initial', padding='same')(x)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.1')(first_conv)
        tmp = Activation('relu')(tmp)
        z1 = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.1',
                    padding='same')(tmp)

        c11 = Conv3D(8, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_conn_1.1',
                     padding='same')(x)
        end_11 = Add()([z1, c11])

        # First_b block (down)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.2_b')(end_11)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.2_b',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.3_b')(tmp)
        tmp = Activation('relu')(tmp)
        z1_b = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.3_b',
                      padding='same')(tmp)

        end_11_b = Add()([z1_b, end_11])

        # First_d block (down)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.2_c')(end_11_b)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.2_c',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.3_c')(tmp)
        tmp = Activation('relu')(tmp)
        z1_c = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.3_c',
                      padding='same')(tmp)

        end_11_c = Add()([z1_c, end_11_b])

        # First_c block (down)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.2_d')(end_11_c)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.2_d',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.3_d')(tmp)
        tmp = Activation('relu')(tmp)
        z1_d = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.3_d',
                      padding='same')(tmp)

        end_11_d = Add()([z1_d, end_11_c])

        # First_d block (down)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.2_e')(end_11_d)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.2_e',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.3_e')(tmp)
        tmp = Activation('relu')(tmp)
        z1_e = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.3_e',
                      padding='same')(tmp)

        end_11_e = Add()([z1_e, end_11_d])

        # First_f block (down)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.2_f')(end_11_e)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.2_f',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.3_f')(tmp)
        tmp = Activation('relu')(tmp)
        z1_f = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.3_f',
                      padding='same')(tmp)

        end_11_f = Add()([z1_f, end_11_e])


        tmp = BatchNormalization(axis=4, name='batch_norm_1.7')(end_11_f)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_pre_softmax', padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_pre_softmax')(tmp)
        in_softmax = Activation('relu')(tmp)

        # Classification layer
        if shortcut_input:
            in_softmax = Concatenate(axis=4)([in_softmax, x])

        classification = Conv3D(num_classes, (1, 1, 1), kernel_initializer=initializer,
                                name='final_convolution_1x1x1')(in_softmax)

        true_mask = Input(shape=segment_dimensions + (1,), name='V-net_mask1')

        mask1 = Lambda(repeat_channels(num_classes), output_shape=repeat_channels_shape(num_classes),
                       name='repeat_mask')(true_mask)

        cmp_mask = Lambda(complementary_mask, output_shape=segment_dimensions + (1,),
                          name='complementary_tumor_mask')(true_mask)
        cmp_mask = Concatenate()([cmp_mask,
                                  Lambda(lambda x: K.zeros_like(x))(cmp_mask),
                                  Lambda(lambda x: K.zeros_like(x))(cmp_mask),
                                  Lambda(lambda x: K.zeros_like(x))(cmp_mask),
                                  ])

        y = softmax_activation(classification)
        y = Multiply(name='mask_background')([y, mask1])
        y = Add(name='label_background')([y, cmp_mask])

        model = Model(inputs=[x, true_mask], outputs=y)

        return model, output_shape

    @staticmethod
    def iSeg_ACNN(num_modalities, segment_dimensions, num_classes, CAE, shortcut_input=False,
                 l1=0.0, l2=0.0):
        """
        U-Net based architecture for segmentation (http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)
            - Convolutional layers: 3x3x3 filters, stride 1, same border
            - Max pooling: 2x2x2
            - Upsampling layers: 2x2x2
            - Activations: ReLU
            - Classification layer: 1x1x1 kernel, and there are as many kernels as classes to be predicted. Element-wise
              softmax as activation.
            - Loss: 3D categorical cross-entropy
            - Optimizer: Adam with lr=0.001, beta_1=0.9, beta_2=0.999 and epsilon=10 ** (-4)
            - Regularization: L1=0.000001, L2=0.0001
            - Weights initialization: He et al normal initialization (https://arxiv.org/abs/1502.01852)

        Returns
        -------
        keras.model
            Compiled Sequential model
        tuple (dim1, dim2, dim3)
            Output shape computed from segment_dimensions and the convolutional architecture
        """
        if not isinstance(segment_dimensions, tuple) or len(segment_dimensions) != 3:
            raise ValueError('segment_dimensions must be a tuple with length 3, specifying the shape of the '
                             'input segment')

        # for dim in segment_dimensions:
        #     assert dim % 32 == 0  # As there are 5 (2, 2, 2) max-poolings, 2 ** 5 is the minimum input size

        # Hyperaparametre values

        initializer = 'he_normal'
        pool_size = (2, 2, 2)

        # Compute input shape, receptive field and output shape after softmax activation
        input_shape = segment_dimensions + (num_modalities,)
        output_shape = segment_dimensions

        # Activations, regularizers and optimizers
        softmax_activation = Lambda(elementwise_softmax_3d, name='Softmax')
        regularizer = l1_l2(l1=l1, l2=l2)

        # Architecture definition
        # INPUT
        x = Input(shape=input_shape, name='V-net_input')

        # First block (down)
        first_conv = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer,
                            name='conv_initial', padding='same')(x)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.1')(first_conv)
        tmp = Activation('relu')(tmp)
        z1 = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.1',
                    padding='same')(tmp)

        c11 = Conv3D(8, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_conn_1.1',
                     padding='same')(x)
        end_11 = Add()([z1, c11])

        # First_b block (down)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.2_b')(end_11)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.2_b',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.3_b')(tmp)
        tmp = Activation('relu')(tmp)
        z1_b = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.3_b',
                      padding='same')(tmp)

        end_11_b = Add()([z1_b, end_11])

        # First_d block (down)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.2_c')(end_11_b)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.2_c',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.3_c')(tmp)
        tmp = Activation('relu')(tmp)
        z1_c = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.3_c',
                      padding='same')(tmp)

        end_11_c = Add()([z1_c, end_11_b])

        # First_c block (down)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.2_d')(end_11_c)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.2_d',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.3_d')(tmp)
        tmp = Activation('relu')(tmp)
        z1_d = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.3_d',
                      padding='same')(tmp)

        end_11_d = Add()([z1_d, end_11_c])

        # Second block (down)
        tmp = BatchNormalization(axis=4, name='batch_norm_2.1')(end_11)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(16, (2, 2, 2), strides=(2, 2, 2), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='downpool_1', padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_2.2')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_2.1',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_2.3')(tmp)
        tmp = Activation('relu')(tmp)
        z2 = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_2.2',
                    padding='same')(tmp)

        c21 = MaxPooling3D(pool_size=pool_size, name='pool_1')(end_11)
        c21 = Conv3D(16, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_conn_2.1', padding='same')(c21)

        end_21 = Add()([z2, c21])

        # Third block (down)
        tmp = BatchNormalization(axis=4, name='batch_norm_3.1')(end_21)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(32, (2, 2, 2), strides=(2, 2, 2), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='downpool_2', padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_3.2')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_3.1',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_3.3')(tmp)
        tmp = Activation('relu')(tmp)
        z3 = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_3.2',
                    padding='same')(tmp)

        c31 = MaxPooling3D(pool_size=pool_size, name='pool_2')(end_21)
        c31 = Conv3D(32, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_conn_3.1', padding='same')(c31)

        end_31 = Add()([z3, c31])

        # Fourth block (down)
        tmp = BatchNormalization(axis=4, name='batch_norm_4.1')(end_31)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(64, (2, 2, 2), strides=(2, 2, 2), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='downpool_3', padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_4.2')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_4.1',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_4.3')(tmp)
        tmp = Activation('relu')(tmp)
        z4 = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_4.2',
                    padding='same')(tmp)

        c41 = MaxPooling3D(pool_size=pool_size, name='pool_3')(end_31)
        c41 = Conv3D(64, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_conn_4.1', padding='same')(c41)

        end_41 = Add()([z4, c41])

        # Fifth block
        tmp = BatchNormalization(axis=4, name='batch_norm_5.1')(end_41)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(128, (2, 2, 2), strides=(2, 2, 2), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='downpool_4', padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_5.2')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(128, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_5.1',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_5.3')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(128, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_5.2',
                     padding='same')(tmp)  # inflection point

        c5 = MaxPooling3D(pool_size=pool_size, name='pool_4')(end_41)
        c5 = Conv3D(128, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_conn_5',
                    padding='same')(c5)

        end_5 = Add()([tmp, c5])

        # Fourth block (up)
        tmp = BatchNormalization(axis=4, name='batch_norm_4.4')(end_5)
        tmp = Activation('relu')(tmp)
        tmp = UpSampling3D(size=pool_size, name='up_4')(tmp)
        tmp = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_up_4',
                     padding='same')(tmp)
        tmp = Concatenate(axis=4)([tmp, z4])
        tmp = BatchNormalization(axis=4, name='batch_norm_4.5')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_4.3',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_4.6')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_4.4',
                     padding='same')(tmp)

        c42 = UpSampling3D(size=pool_size, name='up_4conn')(end_5)
        c42 = Conv3D(64, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_conn_4.2', padding='same')(c42)

        end_42 = Add()([tmp, c42])

        # Third block (up)
        tmp = BatchNormalization(axis=4, name='batch_norm_3.4')(end_42)
        tmp = Activation('relu')(tmp)
        tmp = UpSampling3D(size=pool_size, name='up_3')(tmp)
        tmp = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_up_3',
                     padding='same')(tmp)
        tmp = Concatenate(axis=4)([tmp, z3])
        tmp = BatchNormalization(axis=4, name='batch_norm_3.5')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_3.3',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_3.6')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_3.4',
                     padding='same')(tmp)

        c32 = UpSampling3D(size=pool_size, name='up_3conn')(end_42)
        c32 = Conv3D(32, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_conn_3.2', padding='same')(c32)

        end_32 = Add()([tmp, c32])

        # Second block (up)
        tmp = BatchNormalization(axis=4, name='batch_norm_2.4')(end_32)
        tmp = Activation('relu')(tmp)
        tmp = UpSampling3D(size=pool_size, name='up_2')(tmp)
        tmp = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_up_2',
                     padding='same')(tmp)
        tmp = Concatenate(axis=4)([tmp, z2])
        tmp = BatchNormalization(axis=4, name='batch_norm_2.5')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_2.3',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_2.6')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_2.4',
                     padding='same')(tmp)

        c22 = UpSampling3D(size=pool_size, name='up_2conn')(end_32)
        c22 = Conv3D(16, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_conn_2.2', padding='same')(c22)

        end_22 = Add()([tmp, c22])

        # First block (up)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.4')(end_22)
        tmp = Activation('relu')(tmp)
        tmp = UpSampling3D(size=pool_size, name='up_1')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_up_1',
                     padding='same')(tmp)
        tmp = Concatenate(axis=4)([tmp, z1])
        tmp = BatchNormalization(axis=4, name='batch_norm_1.5')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.3',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.6')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.4',
                     padding='same')(tmp)

        c12 = UpSampling3D(size=pool_size, name='up_1_conn')(end_22)
        c12 = Conv3D(8, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_conn_1.2',
                     padding='same')(c12)

        end_12 = Add()([tmp, c12])

        # Final convolution
        tmp = Concatenate(axis=4)([end_12, end_11_d])
        tmp = BatchNormalization(axis=4, name='batch_norm_1.7')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_pre_softmax', padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_pre_softmax')(tmp)
        in_softmax = Activation('relu')(tmp)

        # Classification layer
        if shortcut_input:
            in_softmax = Concatenate(axis=4)([in_softmax, x])

        classification = Conv3D(num_classes, (1, 1, 1), kernel_initializer=initializer,
                                name='final_convolution_1x1x1')(in_softmax)

        true_mask = Input(shape=segment_dimensions + (1,), name='V-net_mask')

        mask1 = Lambda(repeat_channels(num_classes), output_shape=repeat_channels_shape(num_classes),
                       name='repeat_mask')(true_mask)

        cmp_mask = Lambda(complementary_mask, output_shape=segment_dimensions + (1,),
                          name='complementary_tumor_mask')(true_mask)
        cmp_mask = Concatenate()([cmp_mask,
                                  Lambda(lambda x: K.zeros_like(x))(cmp_mask),
                                  Lambda(lambda x: K.zeros_like(x))(cmp_mask),
                                  Lambda(lambda x: K.zeros_like(x))(cmp_mask)])

        y = softmax_activation(classification)
        y = Multiply(name='mask_background')([y, mask1])
        y = Add(name='label_background')([y, cmp_mask])

        scae_model = SCAE.stacked_CAE_masked(num_classes, segment_dimensions, CAE, pre_train_CAE=False, stddev=0)

        true_labels = Input(shape=segment_dimensions + (num_classes,), name='V-net_labels')

        hidden_labels = scae_model(true_labels)
        hidden_y = scae_model(y)

        loss_labels = Lambda(categorical_crossentropy_3d_lambda, output_shape = (1,))([true_labels,y])
        loss_scae = Lambda(mean_squared_error_lambda, output_shape=(1,))([hidden_labels,hidden_y])

        # model = Model(inputs=[x, true_mask], outputs=[hidden_y])
        model = Model(inputs=[x, true_labels, true_mask], outputs=[loss_labels,loss_scae])
        # model = Model(inputs=[x, true_labels, true_mask], outputs=[loss_labels,loss_scae])

        return model, output_shape


class WMH_models(object):
    """ Interface that allows you to save and load models and their weights """

    # ------------------------------------------------ CONSTANTS ------------------------------------------------ #

    DEFAULT_MODEL = 'v_net'

    # ------------------------------------------------- METHODS ------------------------------------------------- #

    @classmethod
    def get_model(cls, num_modalities, segment_dimensions, num_classes, model_name=None,
                  **kwargs):
        """
        Returns the compiled model specified by its name.
        If no name is given, the default model is returned, which corresponds to the
        hand-picked model that performs the best.

        Parameters
        ----------
        num_modalities : int
            Number of modalities used as input channels
        segment_dimensions : tuple
            Tuple with 3 elements that specify the shape of the input segments
        num_classes : int
            Number of output classes to be predicted in the segmentation
        model_name : [Optional] String
            Name of the model to be returned
        weights_filename: [Optional] String
            Path to the H5 file containing the weights of the trained model. The user must ensure the
            weights correspond to the model to be loaded.

        Returns
        -------
        keras.Model
            Compiled keras model
        tuple
            Tuple with output size (necessary to adjust the ground truth matrix's size),
            or None if output size is the sames as segment_dimensions.

        Raises
        ------
        TypeError
            If model_name is not a String or None
        ValueError
            If the model specified by model_name does not exist
        """
        if model_name is None:
            return cls.get_model(num_modalities, segment_dimensions, num_classes, model_name=cls.DEFAULT_MODEL,
                                 **kwargs)
        if isinstance(model_name, str):
            try:

                model_getter = cls.__dict__[model_name]
                return model_getter.__func__(num_modalities, segment_dimensions, num_classes,
                                             **kwargs)
            except KeyError:
                raise ValueError('Model {} does not exist. Use the class method {} to know the available model names'
                                 .format(model_name, 'get_model_names'))
        else:
            raise TypeError('model_name must be a String/Unicode sequence or None')

    @classmethod
    def get_model_names(cls):
        """
        List of available models' names

        Returns
        -------
        List
            Names of the available model names to be used in get_model(model_name) method
        """

        def filter_functions(attr_name):
            tmp = ('model' not in attr_name) and ('MODEL' not in attr_name)
            return tmp and ('__' not in attr_name) and ('BASE' not in attr_name)

        return filter(filter_functions, cls.__dict__.keys())

    @staticmethod
    def compile(model, lr=0.0005, num_classes=2, loss_name='cross_entropy', optimizer_name='Adam'):
        if optimizer_name == 'Adam':
            beta_1 = 0.9
            beta_2 = 0.999
            epsilon = 10 ** (-8)
            optimizer = Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, clipnorm=1.)
        elif optimizer_name == 'RMSProp':
            rho = 0.9
            epsilon = 10 ** (-8)
            decay = 0.0
            optimizer = RMSprop(lr=lr, rho=rho, epsilon=epsilon, decay=decay)
        elif optimizer_name == 'SGD':
            optimizer = SGD(lr=lr, momentum=0.0, decay=0.0, nesterov=False)
        else:
            raise ValueError("Please, specify a valid optimizer when compiling")

        if num_classes == 2:
            metrics = [accuracy, dice_0, dice_1, recall_0, recall_1, precision_0, precision_1]
        elif num_classes == 3:
            metrics = [accuracy, dice_0, dice_1, dice_2, recall_0, recall_1, recall_2, precision_0, precision_1,
                       precision_2]
        elif num_classes == 4:
            metrics = [accuracy, dice_0, dice_1, dice_2, dice_3, recall_0, recall_1, recall_2, recall_3,
                       precision_0, precision_1, precision_2, precision_3]
        elif num_classes == 5:
            metrics = [accuracy, dice_whole, dice_core, dice_enhance, recall_0, recall_1, recall_2, recall_3, recall_4,
                       precision_0, precision_1, precision_2, precision_3, precision_4]
        else:
            raise ValueError("Please, specify a valid number of classes when compiling")

        if loss_name == 'cross_entropy':
            print('Loss: cross_entropy')
            loss = categorical_crossentropy_3d
        elif loss_name == 'dice':
            print('Loss: dice')
            loss = dice_cost
        else:
            raise ValueError('Please, specify a valid loss function')

        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics

        )

        return model

    @staticmethod
    def compile_masked(model, lr=0.0005):
        beta_1 = 0.9
        beta_2 = 0.999
        epsilon = 10 ** (-8)
        optimizer = Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, clipnorm=1.)

        loss = [lambda y_true, y_pred: y_pred, lambda y_true, y_pred: y_pred]

        model.compile(
            optimizer=optimizer,
            loss=loss,
            loss_weights = [1, 0.0]

        )
        return model

    @staticmethod
    def v_net_BN(num_modalities, segment_dimensions, num_classes, shortcut_input=False,
              l1=0.0, l2=0.0, mode='train'):
        """
        U-Net based architecture for segmentation (http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)
            - Convolutional layers: 3x3x3 filters, stride 1, same border
            - Max pooling: 2x2x2
            - Upsampling layers: 2x2x2
            - Activations: ReLU
            - Classification layer: 1x1x1 kernel, and there are as many kernels as classes to be predicted. Element-wise
              softmax as activation.
            - Loss: 3D categorical cross-entropy
            - Optimizer: Adam with lr=0.001, beta_1=0.9, beta_2=0.999 and epsilon=10 ** (-4)
            - Regularization: L1=0.000001, L2=0.0001
            - Weights initialization: He et al normal initialization (https://arxiv.org/abs/1502.01852)

        Returns
        -------
        keras.model
            Compiled Sequential model
        tuple (dim1, dim2, dim3)
            Output shape computed from segment_dimensions and the convolutional architecture
        """
        if not isinstance(segment_dimensions, tuple) or len(segment_dimensions) != 3:
            raise ValueError('segment_dimensions must be a tuple with length 3, specifying the shape of the '
                             'input segment')

        # for dim in segment_dimensions:
        #     assert dim % 32 == 0  # As there are 5 (2, 2, 2) max-poolings, 2 ** 5 is the minimum input size

        # Hyperaparametre values

        initializer = 'he_normal'
        pool_size = (2, 2, 2)

        # Compute input shape, receptive field and output shape after softmax activation
        input_shape = segment_dimensions + (num_modalities,)
        output_shape = segment_dimensions

        # Activations, regularizers and optimizers
        softmax_activation = Lambda(elementwise_softmax_3d, name='Softmax')
        regularizer = l1_l2(l1=l1, l2=l2)

        # Architecture definition
        # INPUT
        x = Input(shape=input_shape, name='V-net_input')


        # First block (down)
        first_conv = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer,
                            name='conv_initial', padding='same')(x)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.1')(first_conv)
        tmp = Activation('relu')(tmp)
        z1 = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.1',
                    padding='same')(tmp)

        c11 = Conv3D(8, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_conn_1.1',
                     padding='same')(x)
        end_11 = Add()([z1, c11])






        # First_b block (down)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.2_b')(end_11)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.2_b',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.3_b')(tmp)
        tmp = Activation('relu')(tmp)
        z1_b = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.3_b',
                    padding='same')(tmp)


        end_11_b = Add()([z1_b, end_11])




        # First_d block (down)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.2_c')(end_11_b)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.2_c',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.3_c')(tmp)
        tmp = Activation('relu')(tmp)
        z1_c = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.3_c',
                    padding='same')(tmp)


        end_11_c = Add()([z1_c, end_11_b])



        # First_c block (down)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.2_d')(end_11_c)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.2_d',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.3_d')(tmp)
        tmp = Activation('relu')(tmp)
        z1_d = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.3_d',
                    padding='same')(tmp)

        end_11_d = Add()([z1_d, end_11_c])






        # Second block (down)
        tmp = BatchNormalization(axis=4, name='batch_norm_2.1')(end_11)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(16, (2, 2, 2), strides=(2, 2, 2), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='downpool_1', padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_2.2')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_2.1',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_2.3')(tmp)
        tmp = Activation('relu')(tmp)
        z2 = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_2.2',
                    padding='same')(tmp)

        c21 = MaxPooling3D(pool_size=pool_size, name='pool_1')(end_11)
        c21 = Conv3D(16, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_conn_2.1', padding='same')(c21)

        end_21 = Add()([z2, c21])

        # Third block (down)
        tmp = BatchNormalization(axis=4, name='batch_norm_3.1')(end_21)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(32, (2, 2, 2), strides=(2, 2, 2), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='downpool_2', padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_3.2')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_3.1',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_3.3')(tmp)
        tmp = Activation('relu')(tmp)
        z3 = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_3.2',
                    padding='same')(tmp)

        c31 = MaxPooling3D(pool_size=pool_size, name='pool_2')(end_21)
        c31 = Conv3D(32, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_conn_3.1', padding='same')(c31)

        end_31 = Add()([z3, c31])

        # Fourth block (down)
        tmp = BatchNormalization(axis=4, name='batch_norm_4.1')(end_31)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(64, (2, 2, 2), strides=(2, 2, 2), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='downpool_3', padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_4.2')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_4.1',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_4.3')(tmp)
        tmp = Activation('relu')(tmp)
        z4 = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_4.2',
                    padding='same')(tmp)

        c41 = MaxPooling3D(pool_size=pool_size, name='pool_3')(end_31)
        c41 = Conv3D(64, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_conn_4.1', padding='same')(c41)

        end_41 = Add()([z4, c41])

        # Fifth block
        tmp = BatchNormalization(axis=4, name='batch_norm_5.1')(end_41)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(128, (2, 2, 2), strides=(2, 2, 2), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='downpool_4', padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_5.2')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(128, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_5.1',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_5.3')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(128, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_5.2',
                     padding='same')(tmp)  # inflection point

        c5 = MaxPooling3D(pool_size=pool_size, name='pool_4')(end_41)
        c5 = Conv3D(128, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_conn_5',
                    padding='same')(c5)

        end_5 = Add()([tmp, c5])

        # Fourth block (up)
        tmp = BatchNormalization(axis=4, name='batch_norm_4.4')(end_5)
        tmp = Activation('relu')(tmp)
        tmp = UpSampling3D(size=pool_size, name='up_4')(tmp)
        tmp = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_up_4',
                     padding='same')(tmp)
        tmp = Concatenate(axis=4)([tmp, z4])
        tmp = BatchNormalization(axis=4, name='batch_norm_4.5')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_4.3',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_4.6')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_4.4',
                     padding='same')(tmp)

        c42 = UpSampling3D(size=pool_size, name='up_4conn')(end_5)
        c42 = Conv3D(64, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_conn_4.2', padding='same')(c42)

        end_42 = Add()([tmp, c42])

        # Third block (up)
        tmp = BatchNormalization(axis=4, name='batch_norm_3.4')(end_42)
        tmp = Activation('relu')(tmp)
        tmp = UpSampling3D(size=pool_size, name='up_3')(tmp)
        tmp = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_up_3',
                     padding='same')(tmp)
        tmp = Concatenate(axis=4)([tmp, z3])
        tmp = BatchNormalization(axis=4, name='batch_norm_3.5')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_3.3',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_3.6')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_3.4',
                     padding='same')(tmp)

        c32 = UpSampling3D(size=pool_size, name='up_3conn')(end_42)
        c32 = Conv3D(32, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_conn_3.2', padding='same')(c32)

        end_32 = Add()([tmp, c32])

        # Second block (up)
        tmp = BatchNormalization(axis=4, name='batch_norm_2.4')(end_32)
        tmp = Activation('relu')(tmp)
        tmp = UpSampling3D(size=pool_size, name='up_2')(tmp)
        tmp = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_up_2',
                     padding='same')(tmp)
        tmp = Concatenate(axis=4)([tmp, z2])
        tmp = BatchNormalization(axis=4, name='batch_norm_2.5')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_2.3',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_2.6')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_2.4',
                     padding='same')(tmp)

        c22 = UpSampling3D(size=pool_size, name='up_2conn')(end_32)
        c22 = Conv3D(16, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_conn_2.2', padding='same')(c22)

        end_22 = Add()([tmp, c22])

        # First block (up)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.4')(end_22)
        tmp = Activation('relu')(tmp)
        tmp = UpSampling3D(size=pool_size, name='up_1')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_up_1',
                     padding='same')(tmp)
        tmp = Concatenate(axis=4)([tmp, z1])
        tmp = BatchNormalization(axis=4, name='batch_norm_1.5')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.3',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.6')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.4',
                     padding='same')(tmp)

        c12 = UpSampling3D(size=pool_size, name='up_1_conn')(end_22)
        c12 = Conv3D(8, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_conn_1.2',
                     padding='same')(c12)

        end_12 = Add()([tmp, c12])

        # Final convolution

        tmp = BatchNormalization(axis=4, name='batch_norm_1.7')(end_12)
        if shortcut_input:
            tmp = Concatenate(axis=4)([tmp, Lambda(lambda x: x[:, :, :, :, 0:1])(x)])

        tmp = Activation('relu')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_pre_softmax', padding='same')(tmp)

        tmp = BatchNormalization(axis=4, name='batch_norm_pre_softmax')(tmp)

        in_softmax = Activation('relu')(tmp)

        classification = Conv3D(num_classes, (1, 1, 1), kernel_initializer=initializer,
                                name='final_convolution_1x1x1')(in_softmax)

        true_mask = Input(shape=segment_dimensions + (1,), name='V-net_mask1')

        mask1 = Lambda(repeat_channels(num_classes), output_shape=repeat_channels_shape(num_classes),
                       name='repeat_mask')(true_mask)

        cmp_mask = Lambda(complementary_mask, output_shape=segment_dimensions + (1,),
                          name='complementary_tumor_mask')(true_mask)
        cmp_mask = Concatenate()([cmp_mask,
                                  Lambda(lambda x: K.zeros_like(x))(cmp_mask),
                                  Lambda(lambda x: K.zeros_like(x))(cmp_mask)
                                  ])

        y = softmax_activation(classification)
        y = Multiply(name='mask_background')([y, mask1])
        y = Add(name='label_background')([y, cmp_mask])

        model = Model(inputs=[x, true_mask], outputs=y)

        return model, output_shape

    @staticmethod
    def v_net_BN_patches(num_modalities, segment_dimensions, num_classes, shortcut_input=False,
              l1=0.0, l2=0.0, mode='train'):
        """
        U-Net based architecture for segmentation (http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)
            - Convolutional layers: 3x3x3 filters, stride 1, same border
            - Max pooling: 2x2x2
            - Upsampling layers: 2x2x2
            - Activations: ReLU
            - Classification layer: 1x1x1 kernel, and there are as many kernels as classes to be predicted. Element-wise
              softmax as activation.
            - Loss: 3D categorical cross-entropy
            - Optimizer: Adam with lr=0.001, beta_1=0.9, beta_2=0.999 and epsilon=10 ** (-4)
            - Regularization: L1=0.000001, L2=0.0001
            - Weights initialization: He et al normal initialization (https://arxiv.org/abs/1502.01852)

        Returns
        -------
        keras.model
            Compiled Sequential model
        tuple (dim1, dim2, dim3)
            Output shape computed from segment_dimensions and the convolutional architecture
        """
        if not isinstance(segment_dimensions, tuple) or len(segment_dimensions) != 3:
            raise ValueError('segment_dimensions must be a tuple with length 3, specifying the shape of the '
                             'input segment')

        # for dim in segment_dimensions:
        #     assert dim % 32 == 0  # As there are 5 (2, 2, 2) max-poolings, 2 ** 5 is the minimum input size

        # Hyperaparametre values

        initializer = 'he_normal'
        pool_size = (2, 2, 2)

        # Compute input shape, receptive field and output shape after softmax activation
        input_shape = segment_dimensions + (num_modalities,)
        output_shape = segment_dimensions

        # Activations, regularizers and optimizers
        softmax_activation = Lambda(elementwise_softmax_3d, name='Softmax')
        regularizer = l1_l2(l1=l1, l2=l2)

        # Architecture definition
        # INPUT
        x = Input(shape=input_shape, name='V-net_input')


        # First block (down)
        first_conv = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer,
                            name='conv_initial', padding='same')(x)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.1')(first_conv)
        tmp = Activation('relu')(tmp)
        z1 = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.1',
                    padding='same')(tmp)

        c11 = Conv3D(8, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_conn_1.1',
                     padding='same')(x)
        end_11 = Add()([z1, c11])






        # First_b block (down)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.2_b')(end_11)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.2_b',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.3_b')(tmp)
        tmp = Activation('relu')(tmp)
        z1_b = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.3_b',
                    padding='same')(tmp)


        end_11_b = Add()([z1_b, end_11])




        # First_d block (down)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.2_c')(end_11_b)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.2_c',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.3_c')(tmp)
        tmp = Activation('relu')(tmp)
        z1_c = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.3_c',
                    padding='same')(tmp)


        end_11_c = Add()([z1_c, end_11_b])



        # First_c block (down)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.2_d')(end_11_c)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.2_d',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.3_d')(tmp)
        tmp = Activation('relu')(tmp)
        z1_d = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.3_d',
                    padding='same')(tmp)

        end_11_d = Add()([z1_d, end_11_c])






        # Second block (down)
        tmp = BatchNormalization(axis=4, name='batch_norm_2.1')(end_11)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(16, (2, 2, 2), strides=(2, 2, 2), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='downpool_1', padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_2.2')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_2.1',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_2.3')(tmp)
        tmp = Activation('relu')(tmp)
        z2 = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_2.2',
                    padding='same')(tmp)

        c21 = MaxPooling3D(pool_size=pool_size, name='pool_1')(end_11)
        c21 = Conv3D(16, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_conn_2.1', padding='same')(c21)

        end_21 = Add()([z2, c21])

        # Third block (down)
        tmp = BatchNormalization(axis=4, name='batch_norm_3.1')(end_21)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(32, (2, 2, 2), strides=(2, 2, 2), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='downpool_2', padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_3.2')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_3.1',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_3.3')(tmp)
        tmp = Activation('relu')(tmp)
        z3 = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_3.2',
                    padding='same')(tmp)

        c31 = MaxPooling3D(pool_size=pool_size, name='pool_2')(end_21)
        c31 = Conv3D(32, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_conn_3.1', padding='same')(c31)

        end_31 = Add()([z3, c31])

        # Fourth block (down)
        tmp = BatchNormalization(axis=4, name='batch_norm_4.1')(end_31)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(64, (2, 2, 2), strides=(2, 2, 2), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='downpool_3', padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_4.2')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_4.1',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_4.3')(tmp)
        tmp = Activation('relu')(tmp)
        z4 = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_4.2',
                    padding='same')(tmp)

        c41 = MaxPooling3D(pool_size=pool_size, name='pool_3')(end_31)
        c41 = Conv3D(64, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_conn_4.1', padding='same')(c41)

        end_41 = Add()([z4, c41])

        # Fifth block
        tmp = BatchNormalization(axis=4, name='batch_norm_5.1')(end_41)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(128, (2, 2, 2), strides=(2, 2, 2), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='downpool_4', padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_5.2')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(128, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_5.1',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_5.3')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(128, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_5.2',
                     padding='same')(tmp)  # inflection point

        c5 = MaxPooling3D(pool_size=pool_size, name='pool_4')(end_41)
        c5 = Conv3D(128, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_conn_5',
                    padding='same')(c5)

        end_5 = Add()([tmp, c5])

        # Fourth block (up)
        tmp = BatchNormalization(axis=4, name='batch_norm_4.4')(end_5)
        tmp = Activation('relu')(tmp)
        tmp = UpSampling3D(size=pool_size, name='up_4')(tmp)
        tmp = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_up_4',
                     padding='same')(tmp)
        tmp = Concatenate(axis=4)([tmp, z4])
        tmp = BatchNormalization(axis=4, name='batch_norm_4.5')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_4.3',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_4.6')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_4.4',
                     padding='same')(tmp)

        c42 = UpSampling3D(size=pool_size, name='up_4conn')(end_5)
        c42 = Conv3D(64, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_conn_4.2', padding='same')(c42)

        end_42 = Add()([tmp, c42])

        # Third block (up)
        tmp = BatchNormalization(axis=4, name='batch_norm_3.4')(end_42)
        tmp = Activation('relu')(tmp)
        tmp = UpSampling3D(size=pool_size, name='up_3')(tmp)
        tmp = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_up_3',
                     padding='same')(tmp)
        tmp = Concatenate(axis=4)([tmp, z3])
        tmp = BatchNormalization(axis=4, name='batch_norm_3.5')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_3.3',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_3.6')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_3.4',
                     padding='same')(tmp)

        c32 = UpSampling3D(size=pool_size, name='up_3conn')(end_42)
        c32 = Conv3D(32, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_conn_3.2', padding='same')(c32)

        end_32 = Add()([tmp, c32])

        # Second block (up)
        tmp = BatchNormalization(axis=4, name='batch_norm_2.4')(end_32)
        tmp = Activation('relu')(tmp)
        tmp = UpSampling3D(size=pool_size, name='up_2')(tmp)
        tmp = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_up_2',
                     padding='same')(tmp)
        tmp = Concatenate(axis=4)([tmp, z2])
        tmp = BatchNormalization(axis=4, name='batch_norm_2.5')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_2.3',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_2.6')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_2.4',
                     padding='same')(tmp)

        c22 = UpSampling3D(size=pool_size, name='up_2conn')(end_32)
        c22 = Conv3D(16, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_conn_2.2', padding='same')(c22)

        end_22 = Add()([tmp, c22])

        # First block (up)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.4')(end_22)
        tmp = Activation('relu')(tmp)
        tmp = UpSampling3D(size=pool_size, name='up_1')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_up_1',
                     padding='same')(tmp)
        tmp = Concatenate(axis=4)([tmp, z1])
        tmp = BatchNormalization(axis=4, name='batch_norm_1.5')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.3',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.6')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.4',
                     padding='same')(tmp)

        c12 = UpSampling3D(size=pool_size, name='up_1_conn')(end_22)
        c12 = Conv3D(8, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_conn_1.2',
                     padding='same')(c12)

        end_12 = Add()([tmp, c12])

        # Final convolution

        tmp = BatchNormalization(axis=4, name='batch_norm_1.7')(end_12)
        if shortcut_input:
            tmp = Concatenate(axis=4)([tmp, Lambda(lambda x: x[:, :, :, :, 0:1])(x)])

        tmp = Activation('relu')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_pre_softmax', padding='same')(tmp)

        tmp = BatchNormalization(axis=4, name='batch_norm_pre_softmax')(tmp)

        in_softmax = Activation('relu')(tmp)

        classification = Conv3D(num_classes, (1, 1, 1), kernel_initializer=initializer,
                                name='final_convolution_1x1x1')(in_softmax)

        y = softmax_activation(classification)

        if mode == 'train':
            model = Model(inputs=[x], outputs=y)

        else:

            true_mask = Input(shape=segment_dimensions + (1,), name='V-net_mask1')
            mask1 = Lambda(repeat_channels(num_classes), output_shape=repeat_channels_shape(num_classes),
                           name='repeat_mask')(true_mask)
            cmp_mask = Lambda(complementary_mask, output_shape=segment_dimensions + (1,),
                              name='complementary_tumor_mask')(true_mask)
            cmp_mask = Concatenate()([cmp_mask,
                                      Lambda(lambda x: K.zeros_like(x))(cmp_mask),
                                      Lambda(lambda x: K.zeros_like(x))(cmp_mask)
                                      ])

            y = Multiply(name='mask_background')([y, mask1])
            y = Add(name='label_background')([y, cmp_mask])

            model = Model(inputs=[x, true_mask], outputs=y)

        return model, output_shape

    @staticmethod
    def v_net_BN_patches_sr(num_modalities, segment_dimensions, num_classes, shortcut_input=False,
                                l1=0.0, l2=0.0, mode='train'):
        """
        U-Net based architecture for segmentation (http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)
            - Convolutional layers: 3x3x3 filters, stride 1, same border
            - Max pooling: 2x2x2
            - Upsampling layers: 2x2x2
            - Activations: ReLU
            - Classification layer: 1x1x1 kernel, and there are as many kernels as classes to be predicted. Element-wise
              softmax as activation.
            - Loss: 3D categorical cross-entropy
            - Optimizer: Adam with lr=0.001, beta_1=0.9, beta_2=0.999 and epsilon=10 ** (-4)
            - Regularization: L1=0.000001, L2=0.0001
            - Weights initialization: He et al normal initialization (https://arxiv.org/abs/1502.01852)

        Returns
        -------
        keras.model
            Compiled Sequential model
        tuple (dim1, dim2, dim3)
            Output shape computed from segment_dimensions and the convolutional architecture
        """
        if not isinstance(segment_dimensions, tuple) or len(segment_dimensions) != 3:
            raise ValueError('segment_dimensions must be a tuple with length 3, specifying the shape of the '
                             'input segment')

        # for dim in segment_dimensions:
        #     assert dim % 32 == 0  # As there are 5 (2, 2, 2) max-poolings, 2 ** 5 is the minimum input size

        # Hyperaparametre values

        initializer = 'he_normal'
        pool_size = (2, 2, 2)

        # Compute input shape, receptive field and output shape after softmax activation
        input_shape = segment_dimensions + (num_modalities,)
        output_shape = segment_dimensions

        # Activations, regularizers and optimizers
        softmax_activation = Lambda(elementwise_softmax_3d, name='Softmax')
        regularizer = l1_l2(l1=l1, l2=l2)

        # Architecture definition
        # INPUT
        x = Input(shape=input_shape, name='V-net_input')

        # First block (down)
        first_conv = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer,
                            name='conv_1.0', padding='same')(x)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.1')(first_conv)
        tmp = Activation('relu')(tmp)
        z1 = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.1',
                    padding='same')(tmp)

        c11 = Conv3D(8, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_conn_1.1',
                     padding='same')(x)
        end_11 = Add()([z1, c11])

        # First_b block (down)
        x_up = UpSampling3D(size=pool_size, name='sr_up')(x)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.2_b')(x_up)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(4, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.2_b',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.3_b')(tmp)
        tmp = Activation('relu')(tmp)
        z1_b = Conv3D(4, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.3_b',
                      padding='same')(tmp)

        x_up = Conv3D(4, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                      name='conv_conn_0.1', padding='same')(x_up)

        end_11_b = Add()([z1_b, x_up])

        # First_c block (down)
        # end_11_b = UpSampling3D(size=pool_size, name='sr_up_up')(end_11_b)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.2_c')(end_11_b)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(4, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.2_c',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.3_c')(tmp)
        tmp = Activation('relu')(tmp)
        z1_c = Conv3D(4, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.3_c',
                      padding='same')(tmp)

        # end_11_b = Conv3D(2, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
        #               name='conv_conn_0.2', padding='same')(end_11_b)
        end_11_c = Add()([z1_c, end_11_b])

        # First_d block (down)
        # end_11_c = MaxPooling3D(pool_size=pool_size)(end_11_c)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.2_d')(end_11_c)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(4, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.2_d',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.3_d')(tmp)
        tmp = Activation('relu')(tmp)
        z1_d = Conv3D(4, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.3_d',
                      padding='same')(tmp)

        # end_11_c = Conv3D(4, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
        #                   name='conv_conn_0.3', padding='same')(end_11_c)

        end_11_d = Add()([z1_d, end_11_c])

        # Second block (down)
        tmp = BatchNormalization(axis=4, name='batch_norm_2.1')(end_11)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(16, (2, 2, 2), strides=(2, 2, 2), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='downpool_1', padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_2.2')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_2.1',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_2.3')(tmp)
        tmp = Activation('relu')(tmp)
        z2 = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_2.2',
                    padding='same')(tmp)

        c21 = MaxPooling3D(pool_size=pool_size, name='pool_1')(end_11)
        c21 = Conv3D(16, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_conn_2.1', padding='same')(c21)

        end_21 = Add()([z2, c21])

        # Third block (down)
        tmp = BatchNormalization(axis=4, name='batch_norm_3.1')(end_21)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(32, (2, 2, 2), strides=(2, 2, 2), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='downpool_2', padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_3.2')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_3.1',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_3.3')(tmp)
        tmp = Activation('relu')(tmp)
        z3 = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_3.2',
                    padding='same')(tmp)

        c31 = MaxPooling3D(pool_size=pool_size, name='pool_2')(end_21)
        c31 = Conv3D(32, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_conn_3.1', padding='same')(c31)

        end_31 = Add()([z3, c31])

        # Fourth block (down)
        tmp = BatchNormalization(axis=4, name='batch_norm_4.1')(end_31)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(64, (2, 2, 2), strides=(2, 2, 2), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='downpool_3', padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_4.2')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_4.1',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_4.3')(tmp)
        tmp = Activation('relu')(tmp)
        z4 = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_4.2',
                    padding='same')(tmp)

        c41 = MaxPooling3D(pool_size=pool_size, name='pool_3')(end_31)
        c41 = Conv3D(64, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_conn_4.1', padding='same')(c41)

        end_41 = Add()([z4, c41])

        # Fifth block
        tmp = BatchNormalization(axis=4, name='batch_norm_5.1')(end_41)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(128, (2, 2, 2), strides=(2, 2, 2), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='downpool_4', padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_5.2')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(128, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_5.1',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_5.3')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(128, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_5.2',
                     padding='same')(tmp)  # inflection point

        c5 = MaxPooling3D(pool_size=pool_size, name='pool_4')(end_41)
        c5 = Conv3D(128, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_conn_5',
                    padding='same')(c5)

        end_5 = Add()([tmp, c5])

        # Fourth block (up)
        tmp = BatchNormalization(axis=4, name='batch_norm_4.4')(end_5)
        tmp = Activation('relu')(tmp)
        tmp = UpSampling3D(size=pool_size, name='up_4')(tmp)
        tmp = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_up_4',
                     padding='same')(tmp)
        tmp = Concatenate(axis=4)([tmp, z4])
        tmp = BatchNormalization(axis=4, name='batch_norm_4.5')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_4.3',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_4.6')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_4.4',
                     padding='same')(tmp)

        c42 = UpSampling3D(size=pool_size, name='up_4conn')(end_5)
        c42 = Conv3D(64, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_conn_4.2', padding='same')(c42)

        end_42 = Add()([tmp, c42])

        # Third block (up)
        tmp = BatchNormalization(axis=4, name='batch_norm_3.4')(end_42)
        tmp = Activation('relu')(tmp)
        tmp = UpSampling3D(size=pool_size, name='up_3')(tmp)
        tmp = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_up_3',
                     padding='same')(tmp)
        tmp = Concatenate(axis=4)([tmp, z3])
        tmp = BatchNormalization(axis=4, name='batch_norm_3.5')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_3.3',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_3.6')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_3.4',
                     padding='same')(tmp)

        c32 = UpSampling3D(size=pool_size, name='up_3conn')(end_42)
        c32 = Conv3D(32, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_conn_3.2', padding='same')(c32)

        end_32 = Add()([tmp, c32])

        # Second block (up)
        tmp = BatchNormalization(axis=4, name='batch_norm_2.4')(end_32)
        tmp = Activation('relu')(tmp)
        tmp = UpSampling3D(size=pool_size, name='up_2')(tmp)
        tmp = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_up_2',
                     padding='same')(tmp)
        tmp = Concatenate(axis=4)([tmp, z2])
        tmp = BatchNormalization(axis=4, name='batch_norm_2.5')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_2.3',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_2.6')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_2.4',
                     padding='same')(tmp)

        c22 = UpSampling3D(size=pool_size, name='up_2conn')(end_32)
        c22 = Conv3D(16, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_conn_2.2', padding='same')(c22)

        end_22 = Add()([tmp, c22])

        # First block (up)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.4')(end_22)
        tmp = Activation('relu')(tmp)
        tmp = UpSampling3D(size=pool_size, name='up_1')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_up_1',
                     padding='same')(tmp)
        tmp = Concatenate(axis=4)([tmp, z1])
        tmp = BatchNormalization(axis=4, name='batch_norm_1.5')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.3',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.6')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.4',
                     padding='same')(tmp)

        c12 = UpSampling3D(size=pool_size, name='up_1_conn')(end_22)
        c12 = Conv3D(8, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_conn_1.2',
                     padding='same')(c12)

        end_12 = Add()([tmp, c12])

        # Final convolution
        tmp = MaxPooling3D(pool_size=pool_size, name='sr_down')(end_11_d)
        tmp = Concatenate(axis=4)([end_12, tmp])

        if shortcut_input:
            tmp = Concatenate(axis=4)([tmp, Lambda(lambda x: x[:, :, :, :, 0:1])(x)])

        tmp = BatchNormalization(axis=4, name='batch_norm_1.7')(tmp)
        tmp = Activation('relu')(tmp)

        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_pre_softmax', padding='same')(tmp)

        tmp = BatchNormalization(axis=4, name='batch_norm_pre_softmax')(tmp)
        in_softmax = Activation('relu')(tmp)

        classification = Conv3D(num_classes, (1, 1, 1), kernel_initializer=initializer,
                                name='final_convolution_1x1x1')(in_softmax)
        y = softmax_activation(classification)

        if mode == 'train':
            model = Model(inputs=[x], outputs=y)
        else:
            true_mask = Input(shape=segment_dimensions + (1,), name='V-net_mask1')
            mask1 = Lambda(repeat_channels(num_classes), output_shape=repeat_channels_shape(num_classes),
                           name='repeat_mask')(true_mask)
            cmp_mask = Lambda(complementary_mask, output_shape=segment_dimensions + (1,),
                              name='complementary_tumor_mask')(true_mask)
            cmp_mask = Concatenate()([cmp_mask,
                                      Lambda(lambda x: K.zeros_like(x))(cmp_mask),
                                      # Lambda(lambda x: K.zeros_like(x))(cmp_mask)
                                      ])

            y = Multiply(name='mask_background')([y, mask1])
            y = Add(name='label_background')([y, cmp_mask])

            model = Model(inputs=[x, true_mask], outputs=y)

        return model, output_shape

    @staticmethod
    def v_net_BN_patches_sr_old(num_modalities, segment_dimensions, num_classes, shortcut_input=False,
              l1=0.0, l2=0.0, mode='train'):
        """
        U-Net based architecture for segmentation (http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)
            - Convolutional layers: 3x3x3 filters, stride 1, same border
            - Max pooling: 2x2x2
            - Upsampling layers: 2x2x2
            - Activations: ReLU
            - Classification layer: 1x1x1 kernel, and there are as many kernels as classes to be predicted. Element-wise
              softmax as activation.
            - Loss: 3D categorical cross-entropy
            - Optimizer: Adam with lr=0.001, beta_1=0.9, beta_2=0.999 and epsilon=10 ** (-4)
            - Regularization: L1=0.000001, L2=0.0001
            - Weights initialization: He et al normal initialization (https://arxiv.org/abs/1502.01852)

        Returns
        -------
        keras.model
            Compiled Sequential model
        tuple (dim1, dim2, dim3)
            Output shape computed from segment_dimensions and the convolutional architecture
        """
        if not isinstance(segment_dimensions, tuple) or len(segment_dimensions) != 3:
            raise ValueError('segment_dimensions must be a tuple with length 3, specifying the shape of the '
                             'input segment')

        # for dim in segment_dimensions:
        #     assert dim % 32 == 0  # As there are 5 (2, 2, 2) max-poolings, 2 ** 5 is the minimum input size

        # Hyperaparametre values

        initializer = 'he_normal'
        pool_size = (2, 2, 2)

        # Compute input shape, receptive field and output shape after softmax activation
        input_shape = segment_dimensions + (num_modalities,)
        output_shape = segment_dimensions

        # Activations, regularizers and optimizers
        softmax_activation = Lambda(elementwise_softmax_3d, name='Softmax')
        regularizer = l1_l2(l1=l1, l2=l2)

        # Architecture definition
        # INPUT
        x = Input(shape=input_shape, name='V-net_input')


        # First block (down)
        first_conv = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer,
                            name='conv_1.0', padding='same')(x)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.1')(first_conv)
        tmp = Activation('relu')(tmp)
        z1 = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.1',
                    padding='same')(tmp)

        c11 = Conv3D(8, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_conn_1.1',
                     padding='same')(x)
        end_11 = Add()([z1, c11])






        # First_b block (down)
        x_up = UpSampling3D(size=pool_size, name='sr_up')(x)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.2_b')(x_up)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(4, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.2_b',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.3_b')(tmp)
        tmp = Activation('relu')(tmp)
        z1_b = Conv3D(4, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.3_b',
                    padding='same')(tmp)

        x_up = Conv3D(4, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                      name='conv_conn_0.1', padding='same')(x_up)

        end_11_b = Add()([z1_b, x_up])




        # First_c block (down)
        # end_11_b = UpSampling3D(size=pool_size, name='sr_up_up')(end_11_b)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.2_c')(end_11_b)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(4, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.2_c',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.3_c')(tmp)
        tmp = Activation('relu')(tmp)
        z1_c = Conv3D(4, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.3_c',
                    padding='same')(tmp)

        # end_11_b = Conv3D(2, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
        #               name='conv_conn_0.2', padding='same')(end_11_b)
        end_11_c = Add()([z1_c, end_11_b])



        # First_d block (down)
        # end_11_c = MaxPooling3D(pool_size=pool_size)(end_11_c)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.2_d')(end_11_c)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(4, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.2_d',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.3_d')(tmp)
        tmp = Activation('relu')(tmp)
        z1_d = Conv3D(4, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.3_d',
                    padding='same')(tmp)

        # end_11_c = Conv3D(4, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
        #                   name='conv_conn_0.3', padding='same')(end_11_c)

        end_11_d = Add()([z1_d, end_11_c])






        # Second block (down)
        tmp = BatchNormalization(axis=4, name='batch_norm_2.1')(end_11)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(16, (2, 2, 2), strides=(2, 2, 2), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='downpool_1', padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_2.2')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_2.1',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_2.3')(tmp)
        tmp = Activation('relu')(tmp)
        z2 = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_2.2',
                    padding='same')(tmp)

        c21 = MaxPooling3D(pool_size=pool_size, name='pool_1')(end_11)
        c21 = Conv3D(16, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_conn_2.1', padding='same')(c21)

        end_21 = Add()([z2, c21])

        # Third block (down)
        tmp = BatchNormalization(axis=4, name='batch_norm_3.1')(end_21)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(32, (2, 2, 2), strides=(2, 2, 2), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='downpool_2', padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_3.2')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_3.1',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_3.3')(tmp)
        tmp = Activation('relu')(tmp)
        z3 = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_3.2',
                    padding='same')(tmp)

        c31 = MaxPooling3D(pool_size=pool_size, name='pool_2')(end_21)
        c31 = Conv3D(32, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_conn_3.1', padding='same')(c31)

        end_31 = Add()([z3, c31])

        # Fourth block (down)
        tmp = BatchNormalization(axis=4, name='batch_norm_4.1')(end_31)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(64, (2, 2, 2), strides=(2, 2, 2), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='downpool_3', padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_4.2')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_4.1',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_4.3')(tmp)
        tmp = Activation('relu')(tmp)
        z4 = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_4.2',
                    padding='same')(tmp)

        c41 = MaxPooling3D(pool_size=pool_size, name='pool_3')(end_31)
        c41 = Conv3D(64, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_conn_4.1', padding='same')(c41)

        end_41 = Add()([z4, c41])

        # Fifth block
        tmp = BatchNormalization(axis=4, name='batch_norm_5.1')(end_41)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(128, (2, 2, 2), strides=(2, 2, 2), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='downpool_4', padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_5.2')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(128, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_5.1',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_5.3')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(128, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_5.2',
                     padding='same')(tmp)  # inflection point

        c5 = MaxPooling3D(pool_size=pool_size, name='pool_4')(end_41)
        c5 = Conv3D(128, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_conn_5',
                    padding='same')(c5)

        end_5 = Add()([tmp, c5])

        # Fourth block (up)
        tmp = BatchNormalization(axis=4, name='batch_norm_4.4')(end_5)
        tmp = Activation('relu')(tmp)
        tmp = UpSampling3D(size=pool_size, name='up_4')(tmp)
        tmp = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_up_4',
                     padding='same')(tmp)
        tmp = Concatenate(axis=4)([tmp, z4])
        tmp = BatchNormalization(axis=4, name='batch_norm_4.5')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_4.3',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_4.6')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_4.4',
                     padding='same')(tmp)

        c42 = UpSampling3D(size=pool_size, name='up_4conn')(end_5)
        c42 = Conv3D(64, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_conn_4.2', padding='same')(c42)

        end_42 = Add()([tmp, c42])

        # Third block (up)
        tmp = BatchNormalization(axis=4, name='batch_norm_3.4')(end_42)
        tmp = Activation('relu')(tmp)
        tmp = UpSampling3D(size=pool_size, name='up_3')(tmp)
        tmp = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_up_3',
                     padding='same')(tmp)
        tmp = Concatenate(axis=4)([tmp, z3])
        tmp = BatchNormalization(axis=4, name='batch_norm_3.5')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_3.3',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_3.6')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_3.4',
                     padding='same')(tmp)

        c32 = UpSampling3D(size=pool_size, name='up_3conn')(end_42)
        c32 = Conv3D(32, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_conn_3.2', padding='same')(c32)

        end_32 = Add()([tmp, c32])

        # Second block (up)
        tmp = BatchNormalization(axis=4, name='batch_norm_2.4')(end_32)
        tmp = Activation('relu')(tmp)
        tmp = UpSampling3D(size=pool_size, name='up_2')(tmp)
        tmp = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_up_2',
                     padding='same')(tmp)
        tmp = Concatenate(axis=4)([tmp, z2])
        tmp = BatchNormalization(axis=4, name='batch_norm_2.5')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_2.3',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_2.6')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_2.4',
                     padding='same')(tmp)

        c22 = UpSampling3D(size=pool_size, name='up_2conn')(end_32)
        c22 = Conv3D(16, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_conn_2.2', padding='same')(c22)

        end_22 = Add()([tmp, c22])

        # First block (up)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.4')(end_22)
        tmp = Activation('relu')(tmp)
        tmp = UpSampling3D(size=pool_size, name='up_1')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_up_1',
                     padding='same')(tmp)
        tmp = Concatenate(axis=4)([tmp, z1])
        tmp = BatchNormalization(axis=4, name='batch_norm_1.5')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.3',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.6')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.4',
                     padding='same')(tmp)

        c12 = UpSampling3D(size=pool_size, name='up_1_conn')(end_22)
        c12 = Conv3D(8, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_conn_1.2',
                     padding='same')(c12)

        end_12 = Add()([tmp, c12])

        # Final convolution
        tmp = MaxPooling3D(pool_size=pool_size, name = 'sr_down')(end_11_d)
        tmp = Concatenate(axis=4)([end_12, tmp])

        tmp = BatchNormalization(axis=4, name='batch_norm_1.7')(tmp)
        tmp = Activation('relu')(tmp)

        if shortcut_input:
            tmp = Concatenate(axis=4)([tmp, Lambda(lambda x: x[:, :, :, :, 0:1])(x)])

        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_pre_softmax', padding='same')(tmp)

        tmp = BatchNormalization(axis=4, name='batch_norm_pre_softmax')(tmp)
        in_softmax = Activation('relu')(tmp)

        classification = Conv3D(num_classes, (1, 1, 1), kernel_initializer=initializer,
                                name='final_convolution_1x1x1')(in_softmax)
        y = softmax_activation(classification)

        if mode == 'train':
            model = Model(inputs=[x], outputs=y)
        else:
            true_mask = Input(shape=segment_dimensions + (1,), name='V-net_mask1')
            mask1 = Lambda(repeat_channels(num_classes), output_shape=repeat_channels_shape(num_classes),
                           name='repeat_mask')(true_mask)
            cmp_mask = Lambda(complementary_mask, output_shape=segment_dimensions + (1,),
                              name='complementary_tumor_mask')(true_mask)
            cmp_mask = Concatenate()([cmp_mask,
                                      Lambda(lambda x: K.zeros_like(x))(cmp_mask),
                                      Lambda(lambda x: K.zeros_like(x))(cmp_mask)
                                      ])

            y = Multiply(name='mask_background')([y, mask1])
            y = Add(name='label_background')([y, cmp_mask])

            model = Model(inputs=[x, true_mask], outputs=y)

        return model, output_shape


    @staticmethod
    def v_net_BN_masked(num_modalities, segment_dimensions, num_classes, shortcut_input=False,
                        l1=0.0, l2=0.0):
        """
        U-Net based architecture for segmentation (http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)
            - Convolutional layers: 3x3x3 filters, stride 1, same border
            - Max pooling: 2x2x2
            - Upsampling layers: 2x2x2
            - Activations: ReLU
            - Classification layer: 1x1x1 kernel, and there are as many kernels as classes to be predicted. Element-wise
              softmax as activation.
            - Loss: 3D categorical cross-entropy
            - Optimizer: Adam with lr=0.001, beta_1=0.9, beta_2=0.999 and epsilon=10 ** (-4)
            - Regularization: L1=0.000001, L2=0.0001
            - Weights initialization: He et al normal initialization (https://arxiv.org/abs/1502.01852)

        Returns
        -------
        keras.model
            Compiled Sequential model
        tuple (dim1, dim2, dim3)
            Output shape computed from segment_dimensions and the convolutional architecture
        """
        if not isinstance(segment_dimensions, tuple) or len(segment_dimensions) != 3:
            raise ValueError('segment_dimensions must be a tuple with length 3, specifying the shape of the '
                             'input segment')

        # for dim in segment_dimensions:
        #     assert dim % 32 == 0  # As there are 5 (2, 2, 2) max-poolings, 2 ** 5 is the minimum input size

        # Hyperaparametre values

        initializer = 'he_normal'
        pool_size = (2, 2, 2)

        # Compute input shape, receptive field and output shape after softmax activation
        input_shape = segment_dimensions + (num_modalities,)
        output_shape = segment_dimensions

        # Activations, regularizers and optimizers
        softmax_activation = Lambda(elementwise_softmax_3d, name='Softmax')
        regularizer = l1_l2(l1=l1, l2=l2)

        # Architecture definition
        # INPUT
        x = Input(shape=input_shape, name='V-net_input')

        # First block (down)
        first_conv = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer,
                            name='conv_initial', padding='same')(x)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.1')(first_conv)
        tmp = Activation('relu')(tmp)
        z1 = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.1',
                    padding='same')(tmp)

        c11 = Conv3D(8, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_conn_1.1',
                     padding='same')(x)
        end_11 = Add()([z1, c11])

        # First_b block (down)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.2_b')(end_11)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.2_b',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.3_b')(tmp)
        tmp = Activation('relu')(tmp)
        z1_b = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.3_b',
                      padding='same')(tmp)

        end_11_b = Add()([z1_b, end_11])

        # First_d block (down)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.2_c')(end_11_b)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.2_c',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.3_c')(tmp)
        tmp = Activation('relu')(tmp)
        z1_c = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.3_c',
                      padding='same')(tmp)

        end_11_c = Add()([z1_c, end_11_b])

        # First_c block (down)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.2_d')(end_11_c)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.2_d',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.3_d')(tmp)
        tmp = Activation('relu')(tmp)
        z1_d = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.3_d',
                      padding='same')(tmp)

        end_11_d = Add()([z1_d, end_11_c])

        # Second block (down)
        tmp = BatchNormalization(axis=4, name='batch_norm_2.1')(end_11)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(16, (2, 2, 2), strides=(2, 2, 2), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='downpool_1', padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_2.2')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_2.1',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_2.3')(tmp)
        tmp = Activation('relu')(tmp)
        z2 = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_2.2',
                    padding='same')(tmp)

        c21 = MaxPooling3D(pool_size=pool_size, name='pool_1')(end_11)
        c21 = Conv3D(16, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_conn_2.1', padding='same')(c21)

        end_21 = Add()([z2, c21])

        # Third block (down)
        tmp = BatchNormalization(axis=4, name='batch_norm_3.1')(end_21)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(32, (2, 2, 2), strides=(2, 2, 2), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='downpool_2', padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_3.2')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_3.1',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_3.3')(tmp)
        tmp = Activation('relu')(tmp)
        z3 = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_3.2',
                    padding='same')(tmp)

        c31 = MaxPooling3D(pool_size=pool_size, name='pool_2')(end_21)
        c31 = Conv3D(32, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_conn_3.1', padding='same')(c31)

        end_31 = Add()([z3, c31])

        # Fourth block (down)
        tmp = BatchNormalization(axis=4, name='batch_norm_4.1')(end_31)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(64, (2, 2, 2), strides=(2, 2, 2), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='downpool_3', padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_4.2')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_4.1',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_4.3')(tmp)
        tmp = Activation('relu')(tmp)
        z4 = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_4.2',
                    padding='same')(tmp)

        c41 = MaxPooling3D(pool_size=pool_size, name='pool_3')(end_31)
        c41 = Conv3D(64, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_conn_4.1', padding='same')(c41)

        end_41 = Add()([z4, c41])

        # Fifth block
        tmp = BatchNormalization(axis=4, name='batch_norm_5.1')(end_41)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(128, (2, 2, 2), strides=(2, 2, 2), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='downpool_4', padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_5.2')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(128, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_5.1',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_5.3')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(128, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_5.2',
                     padding='same')(tmp)  # inflection point

        c5 = MaxPooling3D(pool_size=pool_size, name='pool_4')(end_41)
        c5 = Conv3D(128, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_conn_5',
                    padding='same')(c5)

        end_5 = Add()([tmp, c5])

        # Fourth block (up)
        tmp = BatchNormalization(axis=4, name='batch_norm_4.4')(end_5)
        tmp = Activation('relu')(tmp)
        tmp = UpSampling3D(size=pool_size, name='up_4')(tmp)
        tmp = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_up_4',
                     padding='same')(tmp)
        tmp = Concatenate(axis=4)([tmp, z4])
        tmp = BatchNormalization(axis=4, name='batch_norm_4.5')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_4.3',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_4.6')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_4.4',
                     padding='same')(tmp)

        c42 = UpSampling3D(size=pool_size, name='up_4conn')(end_5)
        c42 = Conv3D(64, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_conn_4.2', padding='same')(c42)

        end_42 = Add()([tmp, c42])

        # Third block (up)
        tmp = BatchNormalization(axis=4, name='batch_norm_3.4')(end_42)
        tmp = Activation('relu')(tmp)
        tmp = UpSampling3D(size=pool_size, name='up_3')(tmp)
        tmp = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_up_3',
                     padding='same')(tmp)
        tmp = Concatenate(axis=4)([tmp, z3])
        tmp = BatchNormalization(axis=4, name='batch_norm_3.5')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_3.3',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_3.6')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_3.4',
                     padding='same')(tmp)

        c32 = UpSampling3D(size=pool_size, name='up_3conn')(end_42)
        c32 = Conv3D(32, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_conn_3.2', padding='same')(c32)

        end_32 = Add()([tmp, c32])

        # Second block (up)
        tmp = BatchNormalization(axis=4, name='batch_norm_2.4')(end_32)
        tmp = Activation('relu')(tmp)
        tmp = UpSampling3D(size=pool_size, name='up_2')(tmp)
        tmp = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_up_2',
                     padding='same')(tmp)
        tmp = Concatenate(axis=4)([tmp, z2])
        tmp = BatchNormalization(axis=4, name='batch_norm_2.5')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_2.3',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_2.6')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_2.4',
                     padding='same')(tmp)

        c22 = UpSampling3D(size=pool_size, name='up_2conn')(end_32)
        c22 = Conv3D(16, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_conn_2.2', padding='same')(c22)

        end_22 = Add()([tmp, c22])

        # First block (up)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.4')(end_22)
        tmp = Activation('relu')(tmp)
        tmp = UpSampling3D(size=pool_size, name='up_1')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_up_1',
                     padding='same')(tmp)
        tmp = Concatenate(axis=4)([tmp, z1])
        tmp = BatchNormalization(axis=4, name='batch_norm_1.5')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.3',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_1.6')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.4',
                     padding='same')(tmp)

        c12 = UpSampling3D(size=pool_size, name='up_1_conn')(end_22)
        c12 = Conv3D(8, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_conn_1.2',
                     padding='same')(c12)

        end_12 = Add()([tmp, c12])

        # Final convolution
        tmp = Concatenate(axis=4)([end_12, end_11_d])
        tmp = BatchNormalization(axis=4, name='batch_norm_1.7')(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_pre_softmax', padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_pre_softmax')(tmp)
        in_softmax = Activation('relu')(tmp)

        # Classification layer
        if shortcut_input:
            in_softmax = Concatenate(axis=4)([in_softmax, x])

        classification = Conv3D(num_classes, (1, 1, 1), kernel_initializer=initializer,
                                name='final_convolution_1x1x1')(in_softmax)

        true_mask = Input(shape=segment_dimensions + (1,), name='V-net_mask1')

        mask1 = Lambda(repeat_channels(num_classes), output_shape=repeat_channels_shape(num_classes),
                       name='repeat_mask')(true_mask)

        cmp_mask = Lambda(complementary_mask, output_shape=segment_dimensions + (1,),
                          name='complementary_tumor_mask')(true_mask)
        cmp_mask = Concatenate()([cmp_mask,
                                  Lambda(lambda x: K.zeros_like(x))(cmp_mask)
                                  ])

        y = softmax_activation(classification)
        y = Multiply(name='mask_background')([y, mask1])
        y = Add(name='label_background')([y, cmp_mask])

        labels = Input(shape=segment_dimensions + (num_classes,), name='V-net_labels')
        loss = Lambda(categorical_crossentropy_3d_masked, output_shape=(1,), name='masked_seg_loss')(
            [y, true_mask, labels])
        dice1 = Lambda(d1, output_shape=(1,), name='dice1')(
            [y, labels])


        model = Model(inputs=[x, true_mask, labels], outputs=[loss, dice1])

        return model, output_shape

