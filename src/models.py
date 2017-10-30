from keras.layers import  Lambda, AveragePooling3D, MaxPooling3D, PReLU, Input, merge, multiply, Dense, Flatten,\
    Merge, BatchNormalization, Conv3D, Concatenate, Add, Activation, UpSampling3D, Multiply, GaussianNoise, Reshape
from keras.models import Sequential, Model
from keras.optimizers import RMSprop, Adam, SGD, Adadelta
from keras.regularizers import l1_l2
from keras import backend as K
from src.layers import repeat_channels, repeat_channels_shape, complementary_mask, dice_1 as d1, dice_2 as d2, \
    dice_3 as d3, BatchNormalizationMasked, repeat_slices, repeat_slices_shape, Conditional

from src.activations import elementwise_softmax_3d
from src.losses import categorical_crossentropy_3d, scae_mean_squared_error_masked, dice_cost,\
    categorical_crossentropy_3d_masked,mean_squared_error_lambda, categorical_crossentropy_3d_lambda, dice_cost_123, \
    dice_cost_2, dice_cost_3, dice_cost_1, dice_cost_12, categorical_crossentropy_3d_SW
from src.metrics import accuracy, dice_whole, dice_enhance, dice_core, recall_0,recall_1,recall_2,recall_3,\
    recall_4,precision_0,precision_1,precision_2,precision_3, precision_4, dice_0,dice_1, dice_1_2D,dice_2,dice_3, \
    dice_whole_mod, dice_core_mod, dice_enhance_mod, dice
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
            loss = dice_cost_1
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
    """ Interface that allows you to save and load models and their weights

        Different methods:
            brats2017: core network + mask & seg tails. Options: load weights and freeze weights of the core network.
                       It uses masked BN.
            brats2017_BN_normal: core network + mask & seg tails. Options: load weights and freeze weights of the core
                                 network. It uses standard BN


     """

    # ------------------------------------------------ CONSTANTS ------------------------------------------------ #

    DEFAULT_MODEL = 'brats2017'

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
    def compile(model, lr=0.0005, optimizer_name='Adam', model_type='complete', loss_name = 'dice'):

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




        if model_type == 'mask':
            if loss_name == 'dice':
                loss = [dice_cost_1]
                metrics = ['accuracy', dice_1]
                loss_weights = [1]

            elif loss_name =='xentropy':
                loss = categorical_crossentropy_3d
                metrics = ['accuracy', dice_1_2D]
                loss_weights = [1]


                model.compile(
                    optimizer=optimizer,
                    loss=loss,
                    metrics=metrics,
                    loss_weights=loss_weights,
                    sample_weight_mode = 'temporal'
                )

                return model
            else:
                raise ValueError('Please, specify a valid loss_name for model_type=mask')

        elif model_type =='segmentation':
            loss = [categorical_crossentropy_3d]
            loss_weights = [1]
            metrics =  ['accuracy', dice_whole_mod, dice_core_mod, dice_enhance_mod]

        elif model_type == 'survival':
            loss=['mse']
            loss_weights = None
            metrics = None

        elif model_type == 'mask_seg':

            if loss_name == 'xentropy_sw':
                loss = [dice_cost_1, categorical_crossentropy_3d_SW]
            elif loss_name == 'xentropy':
                loss = [dice_cost_1, categorical_crossentropy_3d]

            loss_weights = [1, 1]
            metrics = {'label_mask_background': ['accuracy', dice_1],
                       'label_seg_background': ['accuracy', dice_whole_mod, dice_core_mod, dice_enhance_mod]}

        elif model_type == 'mask_load_seg_all':
            if loss_name == 'xentropy_sw':
                loss = [dice_cost_1, categorical_crossentropy_3d_SW]
            elif loss_name == 'xentropy':
                loss = [dice_cost_1, categorical_crossentropy_3d]
            elif loss_name == 'dice':
                loss = [dice_cost_1, dice_cost_12]
            else:
                raise ValueError('Please, specify a valid loss_name for model_type=mask_seg')

            loss_weights = [1, 1]
            metrics = {'model_2': ['accuracy', dice_1],
                       'label_seg_background': ['accuracy', dice_whole_mod, dice_core_mod, dice_enhance_mod]}

        elif model_type == 'mask_load_seg':
            if loss_name == 'dice':
                loss = [dice_cost_12]
            elif loss_name == 'xentropy_sw':
                loss = [categorical_crossentropy_3d_SW]
            elif loss_name == 'xentropy':
                loss = [categorical_crossentropy_3d]
            else:
                raise ValueError('Please, specify a valid loss_name for model_type=mask_seg')

            loss_weights = [1]
            metrics = ['accuracy', dice_whole_mod, dice_core_mod, dice_enhance_mod]

        elif model_type == 'complete':
            loss = [dice_cost_1, categorical_crossentropy_3d, 'mse']
            loss_weights = [0, 0, 1]
            metrics = ['accuracy']

        elif model_type == 'v_net1_4':
            loss = [dice_cost_1]
            loss_weights = [1]
            metrics = ['accuracy', dice_1]

        elif model_type == 'v_net2':
            loss = [categorical_crossentropy_3d]
            loss_weights = [1]
            metrics = [accuracy, dice_whole, dice_core, dice_enhance]
        else:
            if loss_name == 'xentropy':
                loss = [categorical_crossentropy_3d]
            elif loss_name == 'dice':
                loss = [dice_cost]
            else:
                raise ValueError("Please, specify a valid loss_name")

            loss_weights = [1]
            metrics = [accuracy, dice_whole, dice_core, dice_enhance]

        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            loss_weights=loss_weights
        )

        return model


    @staticmethod
    def brats2017(num_modalities, segment_dimensions, num_classes, model_type = 'mask_seg',
                  l1=0.0, l2=0.0, momentum = 0.99, mode='train', core_weights = None, freeze_core_model = None):
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
        mask_brain = Input(shape=segment_dimensions + (1,), name='V-net_mask1')
        true_mask_tumor = Input(shape=segment_dimensions + (1,))


        mask_brain_1 = MaxPooling3D(pool_size=pool_size)(mask_brain)
        mask_brain_2 = MaxPooling3D(pool_size=pool_size)(mask_brain_1)
        mask_brain_3 = MaxPooling3D(pool_size=pool_size)(mask_brain_2)
        mask_brain_4 = MaxPooling3D(pool_size=pool_size)(mask_brain_3)

        mask_brain_0_0 = Lambda(repeat_channels(8), output_shape=repeat_channels_shape(8),
                              name='mask_brain_0.0')(mask_brain)
        mask_brain_0_2 = Lambda(repeat_channels(9), output_shape=repeat_channels_shape(9),
                                name='mask_brain_0.2')(mask_brain)
        mask_brain_0_1 = Lambda(repeat_channels(16), output_shape=repeat_channels_shape(16),
                              name='mask_brain_0.1')(mask_brain)

        mask_brain_1_0 = Lambda(repeat_channels(16), output_shape=repeat_channels_shape(16),
                              name='mask_brain_1.0')(mask_brain_1)

        mask_brain_1_1 = Lambda(repeat_channels(32), output_shape=repeat_channels_shape(32),
                              name='mask_brain_1.1')(mask_brain_1)

        mask_brain_2_0 = Lambda(repeat_channels(32), output_shape=repeat_channels_shape(32),
                              name='mask_brain_2.0')(mask_brain_2)

        mask_brain_2_1 = Lambda(repeat_channels(64), output_shape=repeat_channels_shape(64),
                              name='mask_brain_2.1')(mask_brain_2)

        mask_brain_3_0 = Lambda(repeat_channels(64), output_shape=repeat_channels_shape(64),
                              name='mask_brain_3.0')(mask_brain_3)
        mask_brain_3_1 = Lambda(repeat_channels(128), output_shape=repeat_channels_shape(128),
                              name='mask_brain_3.1')(mask_brain_3)

        mask_brain_4 = Lambda(repeat_channels(128), output_shape=repeat_channels_shape(128),
                              name='repeat_mask_128')(mask_brain_4)

        first_conv = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer,
                            name='conv_initial', padding='same')(x)

        # First block (down)
        tmp = BatchNormalizationMasked(axis=4, momentum=momentum, name='batch_norm_1.1')([first_conv,mask_brain_0_0])
        tmp = Activation('relu')(tmp)
        z1 = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.1',
                    padding='same')(tmp)

        c11 = Conv3D(8, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_conn_1.1',
                     padding='same')(x)
        end_11 = Add()([z1, c11])

        # Second block (down)
        tmp = BatchNormalizationMasked(axis=4, momentum=momentum, name='batch_norm_2.1')([end_11,mask_brain_0_0])
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(16, (2, 2, 2), strides=(2, 2, 2), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='downpool_1', padding='same')(tmp)
        tmp = BatchNormalizationMasked(axis=4, momentum=momentum, name='batch_norm_2.2')([tmp,mask_brain_1_0])
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_2.1',
                     padding='same')(tmp)
        tmp = BatchNormalizationMasked(axis=4, momentum=momentum, name='batch_norm_2.3')([tmp,mask_brain_1_0])
        tmp = Activation('relu')(tmp)
        z2 = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_2.2',
                    padding='same')(tmp)

        c21 = MaxPooling3D(pool_size=pool_size, name='pool_1')(end_11)
        c21 = Conv3D(16, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_conn_2.1',
                     padding='same')(c21)

        end_21 = Add()([z2, c21])

        # Third block (down)
        tmp = BatchNormalizationMasked(axis=4, momentum=momentum, name='batch_norm_3.1')([end_21,mask_brain_1_0])
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(32, (2, 2, 2), strides=(2, 2, 2), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='downpool_2', padding='same')(tmp)
        tmp = BatchNormalizationMasked(axis=4, momentum=momentum, name='batch_norm_3.2')([tmp,mask_brain_2_0])
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_3.1',
                     padding='same')(tmp)
        tmp = BatchNormalizationMasked(axis=4, momentum=momentum, name='batch_norm_3.3')([tmp,mask_brain_2_0])
        tmp = Activation('relu')(tmp)
        z3 = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_3.2',
                    padding='same')(tmp)

        c31 = MaxPooling3D(pool_size=pool_size, name='pool_2')(end_21)
        c31 = Conv3D(32, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_conn_3.1',
                     padding='same')(c31)

        end_31 = Add()([z3, c31])

        # Fourth block (down)
        tmp = BatchNormalizationMasked(axis=4, momentum=momentum, name='batch_norm_4.1')([end_31,mask_brain_2_0])
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(64, (2, 2, 2), strides=(2, 2, 2), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='downpool_3', padding='same')(tmp)
        tmp = BatchNormalizationMasked(axis=4, momentum=momentum, name='batch_norm_4.2')([tmp,mask_brain_3_0])
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_4.1',
                     padding='same')(tmp)
        tmp = BatchNormalizationMasked(axis=4, momentum=momentum, name='batch_norm_4.3')([tmp,mask_brain_3_0])
        tmp = Activation('relu')(tmp)
        z4 = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_4.2',
                    padding='same')(tmp)

        c41 = MaxPooling3D(pool_size=pool_size, name='pool_3')(end_31)
        c41 = Conv3D(64, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_conn_4.1',
                     padding='same')(c41)

        end_41 = Add()([z4, c41])

        # Fifth block
        tmp = BatchNormalizationMasked(axis=4, momentum=momentum, name='batch_norm_5.1')([end_41,mask_brain_3_0])
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(128, (2, 2, 2), strides=(2, 2, 2), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='downpool_4', padding='same')(tmp)
        tmp = BatchNormalizationMasked(axis=4, momentum=momentum, name='batch_norm_5.2')([tmp,mask_brain_4])
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(128, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_5.1',
                     padding='same')(tmp)
        tmp = BatchNormalizationMasked(axis=4, momentum=momentum, name='batch_norm_5.3')([tmp,mask_brain_4])
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(128, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_5.2',
                     padding='same')(tmp)  # inflection point

        c5 = MaxPooling3D(pool_size=pool_size, name='pool_4')(end_41)
        c5 = Conv3D(128, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_conn_5',
                    padding='same')(c5)

        end_5 = Add()([tmp, c5])

        # Fourth block (up)
        tmp = BatchNormalizationMasked(axis=4, momentum=momentum, name='batch_norm_4.4')([end_5,mask_brain_4])
        tmp = Activation('relu')(tmp)
        tmp = UpSampling3D(size=pool_size, name='up_4')(tmp)
        tmp = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_up_4',
                     padding='same')(tmp)
        tmp = Concatenate(axis=4)([tmp, end_41])
        tmp = BatchNormalizationMasked(axis=4, momentum=momentum, name='batch_norm_4.5')([tmp,mask_brain_3_1])
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_4.3',
                     padding='same')(tmp)
        tmp = BatchNormalizationMasked(axis=4, momentum=momentum, name='batch_norm_4.6')([tmp,mask_brain_3_0])
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_4.4',
                     padding='same')(tmp)

        c42 = UpSampling3D(size=pool_size, name='up_4conn')(end_5)
        c42 = Conv3D(64, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_conn_4.2',
                     padding='same')(c42)

        end_42 = Add()([tmp, c42])

        # Third block (up)
        tmp = BatchNormalizationMasked(axis=4, momentum=momentum, name='batch_norm_3.4')([end_42,mask_brain_3_0])
        tmp = Activation('relu')(tmp)
        tmp = UpSampling3D(size=pool_size, name='up_3')(tmp)
        tmp = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_up_3',
                     padding='same')(tmp)
        tmp = Concatenate(axis=4)([tmp, end_31])
        tmp = BatchNormalizationMasked(axis=4, momentum=momentum, name='batch_norm_3.5')([tmp,mask_brain_2_1])
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_3.3',
                     padding='same')(tmp)
        tmp = BatchNormalizationMasked(axis=4, momentum=momentum, name='batch_norm_3.6')([tmp,mask_brain_2_0])
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_3.4',
                     padding='same')(tmp)

        c32 = UpSampling3D(size=pool_size, name='up_3conn')(end_42)
        c32 = Conv3D(32, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_conn_3.2',
                     padding='same')(c32)

        end_32 = Add()([tmp, c32])

        # Second block (up)
        tmp = BatchNormalizationMasked(axis=4, momentum=momentum, name='batch_norm_2.4')([end_32,mask_brain_2_0])
        tmp = Activation('relu')(tmp)
        tmp = UpSampling3D(size=pool_size, name='up_2')(tmp)
        tmp = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_up_2',
                     padding='same')(tmp)
        tmp = Concatenate(axis=4)([tmp, end_21])
        tmp = BatchNormalizationMasked(axis=4, momentum=momentum, name='batch_norm_2.5')([tmp,mask_brain_1_1])
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_2.3',
                     padding='same')(tmp)
        tmp = BatchNormalizationMasked(axis=4, momentum=momentum, name='batch_norm_2.6')([tmp,mask_brain_1_0])
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_2.4',
                     padding='same')(tmp)

        c22 = UpSampling3D(size=pool_size, name='up_2conn')(end_32)
        c22 = Conv3D(16, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_conn_2.2',
                     padding='same')(c22)

        end_22 = Add()([tmp, c22])

        # First block (up)
        tmp = BatchNormalizationMasked(axis=4, momentum=momentum, name='batch_norm_1.4')([end_22,mask_brain_1_0])
        tmp = Activation('relu')(tmp)
        tmp = UpSampling3D(size=pool_size, name='up_1')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_up_1',
                     padding='same')(tmp)
        tmp = Concatenate(axis=4)([tmp, end_11])
        tmp = BatchNormalizationMasked(axis=4, momentum=momentum, name='batch_norm_1.5')([tmp,mask_brain_0_1])
        tmp = Activation('relu')(tmp)

        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.3',
                     padding='same')(tmp)

        tmp = BatchNormalizationMasked(axis=4, momentum=momentum, name='batch_norm_1.6')([tmp,mask_brain_0_0])
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.4',
                     padding='same')(tmp)

        c12 = UpSampling3D(size=pool_size, name='up_1_conn')(end_22)
        c12 = Conv3D(8, (1, 1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_conn_1.2',
                     padding='same')(c12)

        end_12 = Add()([tmp, c12])


        ############## Mask ################

        def mask_network(input_network, activation = 'softmax', trainable = True):
            # Final convolution
            tmp = BatchNormalizationMasked(axis=4, momentum=momentum, name='batch_norm_1.7',
                                           trainable=trainable)([input_network,mask_brain_0_0])
            tmp = Activation('relu')(tmp)
            tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer,
                         name='conv_pre_softmax', padding='same',trainable=trainable)(tmp)
            tmp = BatchNormalizationMasked(axis=4, momentum=momentum, name='batch_norm_pre_softmax',
                                           trainable=trainable)([tmp,mask_brain_0_0])
            in_softmax = Activation('relu')(tmp)


            if activation =='softmax':
                classification = Conv3D(2, (1, 1, 1), kernel_initializer=initializer,
                                        name='final_convolution_1x1x1',trainable=trainable)(in_softmax)
                y = Lambda(elementwise_softmax_3d, name='Softmax')(classification)
                mask_brain_mult = Concatenate()([mask_brain, mask_brain])
                mask_brain_add = Lambda(complementary_mask, output_shape=segment_dimensions + (1,),
                                     name='complementary_brain_mask')(mask_brain)
                mask_brain_add = Concatenate()([mask_brain_add, Lambda(lambda x: K.zeros_like(x))(mask_brain)])

                y = Multiply(name='mask_background')([y, mask_brain_mult])
                y_mask = Add(name='label_mask_background')([y, mask_brain_add])

            elif activation == 'sigmoid':
                classification = Conv3D(1, (1, 1, 1), kernel_initializer=initializer,
                                        name='final_convolution_1x1x1', trainable=trainable)(in_softmax)
                y = Activation('sigmoid')(classification)
                y_mask = Multiply(name='label_mask_background')([y, mask_brain])

            else:
                raise ValueError('Please, specify a proper activation for mask_network')

            return y_mask

        ############## Tumor ################
        def tumor_network(input_network, mask_tumor, brain_mask):

            bn_mask = Lambda(repeat_channels(8), input_shape = segment_dimensions,output_shape=repeat_channels_shape(8),
                                    name='true_mask_tumor_0.0')(brain_mask)

            # Final convolution
            tmp = BatchNormalizationMasked(axis=4, momentum=momentum, name='batch_norm_1.8')([input_network,bn_mask])
            tmp = Activation('relu')(tmp)
            tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer,
                         name='conv_pre_softmax_seg', padding='same')(tmp)

            tmp = BatchNormalizationMasked(axis=4, momentum=momentum, name='batch_norm_pre_softmax_seg')([tmp,bn_mask])
            in_softmax = Activation('relu')(tmp)
            classification = Conv3D(num_classes, (1, 1, 1), kernel_initializer=initializer,
                                    name='final_convolution_1x1x1_seg')(in_softmax)

            y = Lambda(elementwise_softmax_3d, name='Softmax_seg')(classification)



            mask_tumor_mult = Concatenate()([mask_tumor for _ in range(num_classes)])
            mask_tumor_add = Lambda(complementary_mask, output_shape=segment_dimensions + (1,),
                                 name='complementary_tumor_mask')(mask_tumor)
            mask_tumor_add = Concatenate()([mask_tumor_add] +
                                           [Lambda(lambda x: K.zeros_like(x))(mask_tumor_add) for _ in range(num_classes-1)]
                                          )

            y_seg = Multiply(name='mask_seg')([y, mask_tumor_mult])
            y_seg = Add(name='label_seg_background')([y_seg, mask_tumor_add])

            return y_seg


        if model_type == 'mask':
            core_model = Model(inputs = [x, mask_brain], outputs = [end_12])
            if core_weights is not None:
                core_model.load_weights(core_weights,by_name=True)

            if freeze_core_model:
                core_model.trainable = False

            out_core_model = core_model([x, mask_brain])

            y_mask = mask_network(out_core_model)
            model = Model(inputs = [x, mask_brain], outputs = [y_mask])


        elif model_type == 'segmentation':
            core_model = Model(inputs = [x, mask_brain], outputs = [end_12])
            if core_weights is not None:
                core_model.load_weights(core_weights,by_name=True)

            if freeze_core_model:
                core_model.trainable = False

            out_core_model = core_model([x,mask_brain])
            y_seg = tumor_network(out_core_model)

            model = Model(inputs = [x, mask_brain, true_mask_tumor], outputs = [y_seg])


        elif model_type == 'mask_seg':
            core_model = Model(inputs=[x, mask_brain], outputs=[end_12])
            if core_weights is not None:
                core_model.load_weights(core_weights,by_name=True)

            if freeze_core_model:
                core_model.trainable = False

            out_core_model_0 = core_model([x, mask_brain])
            y_mask = mask_network(out_core_model_0)

            if mode == 'test':
                mask_tumor = Lambda(lambda x: K.expand_dims(tf.floor((x + K.epsilon()) / K.max(x, axis=4, keepdims=True))[:, :, :, :, 1]
                                        , axis=4))(y_mask)

                y_seg = tumor_network(out_core_model_0, mask_tumor, mask_brain)
                model = Model(inputs=[x, mask_brain], outputs=[y_mask, y_seg])
            else:

                y_seg = tumor_network(out_core_model_0, true_mask_tumor, mask_brain)
                model = Model(inputs=[x, mask_brain, true_mask_tumor], outputs=[y_mask, y_seg])

        else:
            raise ValueError('Please, specify a valid model_type')


        return model, output_shape

    @staticmethod
    def brats2017_BN_normal(num_modalities, segment_dimensions, num_classes, model_type = 'mask_seg', l1=0.0, l2=0.0,
                            momentum=0.99, mode='train', core_weights=None, freeze_core_model=None):
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
        mask_brain = Input(shape=segment_dimensions + (1,), name='V-net_mask1')
        true_mask_tumor = Input(shape=segment_dimensions + (1,))


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
        def mask_network(input_network, activation='softmax', trainable=True):
            # Final convolution
            tmp = BatchNormalization(axis=4, momentum=momentum, name='batch_norm_1.7',trainable=trainable)(input_network)
            tmp = Activation('relu')(tmp)
            tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer,
                         name='conv_pre_softmax', padding='same', trainable=trainable)(tmp)
            tmp = BatchNormalization(axis=4, momentum=momentum, name='batch_norm_pre_softmax',trainable=trainable)(tmp)
            in_softmax = Activation('relu')(tmp)

            if activation == 'softmax':
                classification = Conv3D(2, (1, 1, 1), kernel_initializer=initializer,
                                        name='final_convolution_1x1x1', trainable=trainable)(in_softmax)
                y = Lambda(elementwise_softmax_3d, name='Softmax')(classification)
                mask_brain_mult = Concatenate()([mask_brain, mask_brain])
                mask_brain_add = Lambda(complementary_mask, output_shape=segment_dimensions + (1,),
                                        name='complementary_brain_mask')(mask_brain)
                mask_brain_add = Concatenate()([mask_brain_add, Lambda(lambda x: K.zeros_like(x))(mask_brain)])

                y = Multiply(name='mask_background')([y, mask_brain_mult])
                y_mask = Add(name='label_mask_background')([y, mask_brain_add])

            elif activation == 'sigmoid':
                classification = Conv3D(1, (1, 1, 1), kernel_initializer=initializer,
                                        name='final_convolution_1x1x1', trainable=trainable)(in_softmax)
                y = Activation('sigmoid')(classification)
                y_mask = Multiply(name='label_mask_background')([y, mask_brain])

            else:
                raise ValueError('Please, specify a proper activation for mask_network')

            return y_mask

        ############## Tumor ################
        def tumor_network(input_network, mask_tumor):

            # Final convolution
            tmp = BatchNormalization(axis=4, momentum=momentum, name='batch_norm_1.8')(input_network)
            tmp = Activation('relu')(tmp)
            tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer,
                         name='conv_pre_softmax_seg', padding='same')(tmp)

            tmp = BatchNormalization(axis=4, momentum=momentum, name='batch_norm_pre_softmax_seg')(tmp)
            in_softmax = Activation('relu')(tmp)
            classification = Conv3D(num_classes, (1, 1, 1), kernel_initializer=initializer,
                                    name='final_convolution_1x1x1_seg')(in_softmax)

            y = Lambda(elementwise_softmax_3d, name='Softmax_seg')(classification)

            mask_tumor_mult = Concatenate()([mask_tumor for _ in range(num_classes)])
            mask_tumor_add = Lambda(complementary_mask, output_shape=segment_dimensions + (1,),
                                    name='complementary_tumor_mask')(mask_tumor)
            mask_tumor_add = Concatenate()([mask_tumor_add] +
                                           [Lambda(lambda x: K.zeros_like(x))(mask_tumor_add) for _ in
                                            range(num_classes - 1)]
                                           )

            y_seg = Multiply(name='mask_seg')([y, mask_tumor_mult])
            y_seg = Add(name='label_seg_background')([y_seg, mask_tumor_add])

            return y_seg

        if model_type == 'mask':
            core_model = Model(inputs=[x, mask_brain], outputs=[end_12])
            if core_weights is not None:
                core_model.load_weights(core_weights, by_name=True)

            if freeze_core_model:
                core_model.trainable = False

            out_core_model = core_model([x, mask_brain])

            y_mask = mask_network(out_core_model)
            model = Model(inputs=[x, mask_brain], outputs=[y_mask])


        elif model_type == 'segmentation':
            core_model = Model(inputs=[x, mask_brain], outputs=[end_12])
            if core_weights is not None:
                core_model.load_weights(core_weights, by_name=True)

            if freeze_core_model:
                core_model.trainable = False

            out_core_model = core_model([x, mask_brain])
            y_seg = tumor_network(out_core_model)

            model = Model(inputs=[x, mask_brain, true_mask_tumor], outputs=[y_seg])


        elif model_type == 'mask_seg':
            core_model = Model(inputs=[x, mask_brain], outputs=[end_12])
            if core_weights is not None:
                core_model.load_weights(core_weights, by_name=True)

            if freeze_core_model:
                core_model.trainable = False

            out_core_model_0 = core_model([x, mask_brain])
            y_mask = mask_network(out_core_model_0)

            if mode == 'test':
                mask_tumor = Lambda(
                    lambda x: K.expand_dims(tf.floor((x + K.epsilon()) / K.max(x, axis=4, keepdims=True))[:, :, :, :, 1]
                                            , axis=4))(y_mask)

                y_seg = tumor_network(out_core_model_0, mask_tumor, mask_brain)
                model = Model(inputs=[x, mask_brain], outputs=[y_mask, y_seg])
            else:

                y_seg = tumor_network(out_core_model_0, true_mask_tumor, mask_brain)
                model = Model(inputs=[x, mask_brain, true_mask_tumor], outputs=[y_mask, y_seg])

        else:
            raise ValueError('Please, specify a valid model_type')

        return model, output_shape

    @staticmethod
    def v_net_BN(num_modalities, segment_dimensions, num_classes,l1=0.0, l2=0.0, momentum = 0.99):
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
        tmp = Activation('relu')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer,
                     name='conv_pre_softmax', padding='same')(tmp)
        tmp = BatchNormalization(axis=4, name='batch_norm_pre_softmax')(tmp)
        in_softmax = Activation('relu')(tmp)


        classification = Conv3D(num_classes, (1, 1, 1), kernel_initializer=initializer,
                                name='final_convolution_1x1x1')(in_softmax)


        #ROI mask.
        true_mask = Input(shape=segment_dimensions + (1,), name='V-net_mask1')

        mask1 = Lambda(repeat_channels(num_classes), output_shape=repeat_channels_shape(num_classes),
                       name='repeat_mask')(true_mask)

        cmp_mask = Lambda(complementary_mask, output_shape=segment_dimensions + (1,),
                          name='complementary_tumor_mask')(true_mask)

        cmp_mask = Concatenate()([cmp_mask] + [Lambda(lambda x: K.zeros_like(x))(cmp_mask) for _ in range(num_classes-1)])

        y = softmax_activation(classification)
        y = Multiply(name='mask_background')([y, mask1])
        y = Add(name='label_background')([y, cmp_mask])

        model = Model(inputs=[x, true_mask], outputs=y)

        return model, output_shape



class iSeg_models(object):
    """ Interface that allows you to save and load models and their weights.

        Different methods:
            v_net: whole volume without BN
            v_net_BN: whole volume with standard BN
            v_net_BN_masked: whole volume with masekd_BN
            v_net_BN_patches: v_net training by patches with standard BN
            v_net_BN_patches_sr: aumgented v_net training by patches with standard BN and

    """

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
            print('Loss: global_dice')
            loss = dice_cost_123
        else:
            raise ValueError('Please, specify a valid loss function')

        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics

        )

        return model


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
    def augmented_v_net_BN_patches(num_modalities, segment_dimensions, num_classes, shortcut_input=False,
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


class WMH_models(object):
    """ Interface that allows you to save and load models and their weights

        Different methods:
            v_net_BN: whole volume with standard BN
            v_net_BN_masked: whole volume with masekd_BN
            v_net_BN_patches: v_net training by patches with standard BN
            v_net_BN_patches_sr: aumgented v_net training by patches with standard BN and


    """

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
            loss = dice_cost_1
        else:
            raise ValueError('Please, specify a valid loss function')

        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics

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
    def augmented_v_net_BN_patches(num_modalities, segment_dimensions, num_classes, shortcut_input=False,
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


        # First block
        first_conv = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer,
                            name='conv_1.0', padding='same')(x)
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
        tmp = Conv3D(4, (3, 3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.2_c',
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


