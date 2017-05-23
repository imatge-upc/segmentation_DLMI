from keras.layers import  Lambda, AveragePooling3D, MaxPooling3D, PReLU, Input, merge, \
    Merge, BatchNormalization, Conv3D, Concatenate, concatenate ,Convolution3D
from keras.models import Sequential, Model
from keras.optimizers import RMSprop, Adam
from keras.regularizers import L1L2

from src.activations import elementwise_softmax_3d
from src.losses import categorical_crossentropy_3d
from src.layers import Upsampling3D_mod
from src.metrics import accuracy, dice_whole, dice_enhance, dice_core, recall_0,recall_1,recall_2,recall_3,\
    recall_4, precision_0,precision_1,precision_2,precision_3,precision_4



class BratsModels(object):
    """ Interface that allows you to save and load models and their weights """

    # ------------------------------------------------ CONSTANTS ------------------------------------------------ #

    DEFAULT_MODEL = 'u_net'

    # ------------------------------------------------- METHODS ------------------------------------------------- #

    @classmethod
    def get_model(cls, num_modalities, segment_dimensions, num_classes, model_name=None, weights_filename=None,
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
                                 weights_filename=weights_filename, **kwargs)
        if isinstance(model_name, basestring):
            try:

                model_getter = cls.__dict__[model_name]
                return model_getter.__func__(num_modalities, segment_dimensions, num_classes,
                                             weights_filename=weights_filename, **kwargs)
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
    def one_pathway(num_modalities, segment_dimensions, num_classes, weights_filename=None):
        """
        Baseline CNN pathway in deepmedic (https://github.com/Kamnitsask/deepmedic).
            - Convolutional layers: 4 layers with 5x5x5 kernels, stride 1 and no padding.
            - Activations: PReLU (https://arxiv.org/abs/1502.01852)
            - Classification layer: 1x1x1 kernel, and there are as many kernels as classes to be predicted. Element-wise
              softmax as activation.
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

        # Hyperaparametre values
        learning_rate = 0.001
        rho = 0.9
        epsilon = 10 ** (-4)
        L1_reg = 0.000001
        L2_reg = 0.0001
        initializer = 'he_normal'

        # Compute input shape, receptive field and output shape after softmax activation
        input_shape = (num_modalities,) + segment_dimensions
        segment_dimensions_list = list(segment_dimensions)
        output_shape = []
        for value in segment_dimensions_list:
            output_shape.append(value - (5 - 1) * 4)  # 5 is kernel size, 1 is stride, and 4 is the # of conv layers
        output_shape = tuple(output_shape)

        # Activations, regularizers and optimizers
        softmax_activation = Lambda(elementwise_softmax_3d, name='softmax')
        regularizer = L1L2(l1=L1_reg, l2=L2_reg)
        optimizer = RMSprop(learning_rate, rho=rho, epsilon=epsilon)

        # Model
        model = Sequential(name='one_pathway')
        model.add(Convolution3D(30, 3, 3, 3, init=initializer, W_regularizer=regularizer, input_shape=input_shape))
        model.add(Convolution3D(30, 3, 3, 3, init=initializer, W_regularizer=regularizer))
        model.add(PReLU())
        model.add(Convolution3D(40, 3, 3, 3, init=initializer, W_regularizer=regularizer))
        model.add(Convolution3D(40, 3, 3, 3, init=initializer, W_regularizer=regularizer))
        model.add(PReLU())
        model.add(Convolution3D(40, 3, 3, 3, init=initializer, W_regularizer=regularizer))
        model.add(Convolution3D(40, 3, 3, 3, init=initializer, W_regularizer=regularizer))
        model.add(PReLU())
        model.add(Convolution3D(50, 3, 3, 3, init=initializer, W_regularizer=regularizer))
        model.add(Convolution3D(50, 3, 3, 3, init=initializer, W_regularizer=regularizer))
        model.add(PReLU())
        model.add(Convolution3D(num_classes, 1, 1, 1, init=initializer, W_regularizer=regularizer))
        model.add(PReLU())
        model.add(softmax_activation)

        # Compile model
        model.compile(
            optimizer=optimizer,
            loss=categorical_crossentropy_3d,
            metrics=[dice_whole]
        )

        # Load weights if available
        if weights_filename is not None:
            model.load_weights(weights_filename)

        return model, output_shape

    @staticmethod
    def one_pathway_no_downsampling(num_modalities, segment_dimensions, num_classes, weights_filename=None):
        """
        Baseline CNN pathway in deepmedic (https://github.com/Kamnitsask/deepmedic), but maintaining borders and
        therefore
            - Convolutional layers: 4 layers with 5x5x5 kernels, stride 1 and padding.
            - Activations: PReLU (https://arxiv.org/abs/1502.01852)
            - Classification layer: 1x1x1 kernel, and there are as many kernels as classes to be predicted. Element-wise
              softmax as activation.
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

        # Hyperaparametre values
        learning_rate = 0.001
        rho = 0.9
        epsilon = 10 ** (-4)
        L1_reg = 0.000001
        L2_reg = 0.0001
        initializer = 'he_normal'

        # Compute input shape, receptive field and output shape after softmax activation
        input_shape = (num_modalities,) + segment_dimensions
        output_shape = segment_dimensions

        # Activations, regularizers and optimizers
        softmax_activation = Lambda(elementwise_softmax_3d, name='softmax')
        regularizer = L1L2(l1=L1_reg, l2=L2_reg)
        optimizer = RMSprop(learning_rate, rho=rho, epsilon=epsilon)

        # Model
        model = Sequential(name='one_pathway')
        model.add(Convolution3D(30, 5, 5, 5, init=initializer, border_mode='same', W_regularizer=regularizer,
                                input_shape=input_shape))
        model.add(PReLU())
        model.add(Convolution3D(40, 5, 5, 5, init=initializer, border_mode='same', W_regularizer=regularizer))
        model.add(PReLU())
        model.add(Convolution3D(40, 5, 5, 5, init=initializer, border_mode='same', W_regularizer=regularizer))
        model.add(PReLU())
        model.add(Convolution3D(50, 5, 5, 5, init=initializer, border_mode='same', W_regularizer=regularizer))
        model.add(PReLU())
        model.add(Convolution3D(num_classes, 1, 1, 1, init=initializer, W_regularizer=regularizer))
        model.add(PReLU())
        model.add(softmax_activation)

        # Compile model
        model.compile(
            optimizer=optimizer,
            loss=categorical_crossentropy_3d,
            metrics=[dice_whole]
        )

        # Load weights if available
        if weights_filename is not None:
            model.load_weights(weights_filename)

        return model, output_shape

    @staticmethod
    def one_pathway_skipped_upsampling(num_modalities, segment_dimensions, num_classes, weights_filename=None):
        """
        Inspired in Long's architecture for segmentation using fully convolutional nets and skipped layers
        (http://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf)
            - Convolutional layers: 3x3x3 filters, stride 1, same border
            - Max pooling: 2x2x2
            - Upsampling layers: 8x8x8, 16x16x16, 32x32x32
            - Activations: PreLU (https://arxiv.org/abs/1502.01852)
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

        for dim in segment_dimensions:
            assert dim % 32 == 0  # As there are 5 (2, 2, 2) max-poolings, 2 ** 5 is the minimum input size

        # Hyperparametre values
        lr = 0.001
        beta_1 = 0.9
        beta_2 = 0.999
        epsilon = 10 ** (-4)
        L1_reg = 0.000001
        L2_reg = 0.0001
        initializer = 'he_normal'
        pool_size = (2, 2, 2)

        # Compute input shape, receptive field and output shape after softmax activation
        input_shape = (num_modalities,) + segment_dimensions
        output_shape = segment_dimensions

        # Activations, regularizers and optimizers
        softmax_activation = Lambda(elementwise_softmax_3d, name='softmax')
        regularizer = L1L2(l1=L1_reg, l2=L2_reg)
        optimizer = Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)

        # Architecture definition

        # Main pathway
        x = Input(shape=input_shape, name='Long input')

        tmp = Convolution3D(8, 3, 3, 3, init=initializer, border_mode='same', W_regularizer=regularizer)(x)
        tmp = PReLU()(tmp)
        tmp = MaxPooling3D(pool_size=pool_size)(tmp)

        tmp = Convolution3D(16, 3, 3, 3, init=initializer, border_mode='same', W_regularizer=regularizer)(tmp)
        tmp = PReLU()(tmp)
        tmp = MaxPooling3D(pool_size=pool_size)(tmp)

        tmp = Convolution3D(32, 3, 3, 3, init=initializer, border_mode='same', W_regularizer=regularizer)(tmp)
        tmp = PReLU()(tmp)
        pool3 = MaxPooling3D(pool_size=pool_size)(tmp)

        tmp = Convolution3D(64, 3, 3, 3, init=initializer, border_mode='same', W_regularizer=regularizer)(pool3)
        tmp = PReLU()(tmp)
        pool4 = MaxPooling3D(pool_size=pool_size)(tmp)

        tmp = Convolution3D(128, 3, 3, 3, init=initializer, border_mode='same', W_regularizer=regularizer)(pool4)
        tmp = PReLU()(tmp)
        pool5 = MaxPooling3D(pool_size=pool_size)(tmp)
        tmp = Convolution3D(128, 3, 3, 3, init=initializer, border_mode='same', W_regularizer=regularizer)(pool5)
        tmp = PReLU()(tmp)
        fcn_32 = Upsampling3D_mod(size=(32, 32, 32))(tmp)

        # Skipped layers: FCN_8
        tmp = Convolution3D(64, 3, 3, 3, init=initializer, border_mode='same', W_regularizer=regularizer)(pool3)
        tmp = PReLU()(tmp)
        tmp = Convolution3D(64, 3, 3, 3, init=initializer, border_mode='same', W_regularizer=regularizer)(tmp)
        tmp = PReLU()(tmp)
        fcn_8 = Upsampling3D_mod(size=(8, 8, 8))(tmp)

        # Skipped layers: FCN_16
        tmp = Convolution3D(128, 3, 3, 3, init=initializer, border_mode='same', W_regularizer=regularizer)(pool4)
        tmp = PReLU()(tmp)
        fcn_16 = Upsampling3D_mod(size=(16, 16, 16))(tmp)

        # Merge fcn_32, fcn_16, fcn_8
        tmp = merge([fcn_8, fcn_16, fcn_32], mode='concat', concat_axis=1)

        # Classification layer
        classification = Convolution3D(num_classes, 1, 1, 1, init=initializer,
                                       W_regularizer=regularizer)(tmp)
        classification = PReLU()(classification)
        y = softmax_activation(classification)

        # Create and compile model
        model = Model(input=x, output=y)
        model.compile(
            optimizer=optimizer,
            loss=categorical_crossentropy_3d,
            metrics=[dice_whole]
        )

        # Load weights if available
        if weights_filename is not None:
            model.load_weights(weights_filename)

        return model, output_shape

    @staticmethod
    def deepmedic(num_modalities, segment_dimensions, num_classes, weights_filename=None, **kwargs):

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

        receptive_field = 17
        downsampling = 3
        segment_dimensions_up = kwargs['segment_dimensions_up'] if kwargs['segment_dimensions_up'] else (
            segment_dimensions[0] + downsampling - 1 - receptive_field * (downsampling - 1),
            segment_dimensions[1] + downsampling - 1 - receptive_field * (downsampling - 1),
            segment_dimensions[2] + downsampling - 1 - receptive_field * (downsampling - 1))

        for dim in segment_dimensions_up:
            assert dim - receptive_field > 0  # As there are 5 (2, 2, 2) max-poolings, 2 ** 5 is the minimum input size
        for dim in segment_dimensions:
            assert dim / downsampling - receptive_field > 0

        # Hyperaparametre values
        learning_rate = 0.001
        rho = 0.9
        epsilon = 1e-4
        L1_reg = 1e-6
        L2_reg = 1e-4
        initializer = 'he_normal'

        # Compute input shape, receptive field and output shape after softmax activation
        input_shape_up = (num_modalities,) + segment_dimensions_up
        input_shape_down = (num_modalities,) + segment_dimensions
        output_shape = []
        for dim in segment_dimensions_up:
            if dim is None:
                output_shape.append(None)
            output_shape.append(dim - receptive_field + 1)
        output_shape = tuple(output_shape)

        # Activations, regularizers and optimizers
        softmax_activation = Lambda(elementwise_softmax_3d, name='softmax')
        regularizer = L1L2(l1=L1_reg, l2=L2_reg)
        optimizer = RMSprop(learning_rate, rho=rho, epsilon=epsilon)

        # Model
        x = Input(shape=input_shape_down, name='Deepmedic input')

        # Upper pathway
        def upper_pathway():
            halfDimInputLeft = ((segment_dimensions[0] - segment_dimensions_up[0]) / 2 + (
                segment_dimensions[0] - segment_dimensions_up[0]) % 2,
                                (segment_dimensions[1] - segment_dimensions_up[1]) / 2 + (
                                    segment_dimensions[1] - segment_dimensions_up[1]) % 2,
                                (segment_dimensions[2] - segment_dimensions_up[2]) / 2 + (
                                    segment_dimensions[2] - segment_dimensions_up[2]) % 2)

            halfDimInputRight = ((segment_dimensions[0] - segment_dimensions_up[0]) / 2,
                                 (segment_dimensions[1] - segment_dimensions_up[1]) / 2,
                                 (segment_dimensions[2] - segment_dimensions_up[2]) / 2)

            x_up = Lambda(lambda x: x[:, :, halfDimInputLeft[0]:-halfDimInputRight[0],
                                    halfDimInputLeft[1]:-halfDimInputRight[1],
                                    halfDimInputLeft[2]:-halfDimInputRight[2]], output_shape=input_shape_up)(x)

            tmp = Convolution3D(30, 3, 3, 3, init=initializer, W_regularizer=regularizer)(x_up)
            tmp = PReLU()(tmp)
            tmp = BatchNormalization(axis=1)(tmp)

            tmp = Convolution3D(30, 3, 3, 3, init=initializer, W_regularizer=regularizer)(tmp)
            tmp = PReLU()(tmp)
            tmp = BatchNormalization(axis=1)(tmp)

            tmp = Convolution3D(40, 3, 3, 3, init=initializer, W_regularizer=regularizer)(tmp)
            tmp = PReLU()(tmp)
            tmp = BatchNormalization(axis=1)(tmp)

            tmp = Convolution3D(40, 3, 3, 3, init=initializer, W_regularizer=regularizer)(tmp)
            tmp = PReLU()(tmp)
            tmp = BatchNormalization(axis=1)(tmp)

            tmp = Convolution3D(40, 3, 3, 3, init=initializer, W_regularizer=regularizer)(tmp)
            tmp = PReLU()(tmp)
            tmp = BatchNormalization(axis=1)(tmp)

            tmp = Convolution3D(40, 3, 3, 3, init=initializer, W_regularizer=regularizer)(tmp)
            tmp = PReLU()(tmp)
            tmp = BatchNormalization(axis=1)(tmp)

            tmp = Convolution3D(50, 3, 3, 3, init=initializer, W_regularizer=regularizer)(tmp)
            tmp = PReLU()(tmp)
            tmp = BatchNormalization(axis=1)(tmp)

            tmp = Convolution3D(50, 3, 3, 3, init=initializer, W_regularizer=regularizer)(tmp)
            tmp = PReLU()(tmp)
            out_up = BatchNormalization(axis=1)(tmp)

            return out_up

        # Lower pathway
        def lower_pathway():
            x_down = AveragePooling3D((3, 3, 3), strides=(downsampling, downsampling, downsampling))(x)
            tmp = Convolution3D(30, 3, 3, 3, init=initializer, W_regularizer=regularizer)(x_down)
            tmp = PReLU()(tmp)
            tmp = BatchNormalization(axis=1)(tmp)

            tmp = Convolution3D(30, 3, 3, 3, init=initializer, W_regularizer=regularizer)(tmp)
            tmp = PReLU()(tmp)
            tmp = BatchNormalization(axis=1)(tmp)

            tmp = Convolution3D(40, 3, 3, 3, init=initializer, W_regularizer=regularizer)(tmp)
            tmp = PReLU()(tmp)
            tmp = BatchNormalization(axis=1)(tmp)

            tmp = Convolution3D(40, 3, 3, 3, init=initializer, W_regularizer=regularizer)(tmp)
            tmp = PReLU()(tmp)
            tmp = BatchNormalization(axis=1)(tmp)

            tmp = Convolution3D(40, 3, 3, 3, init=initializer, W_regularizer=regularizer)(tmp)
            tmp = PReLU()(tmp)
            tmp = BatchNormalization(axis=1)(tmp)

            tmp = Convolution3D(40, 3, 3, 3, init=initializer, W_regularizer=regularizer)(tmp)
            tmp = PReLU()(tmp)
            tmp = BatchNormalization(axis=1)(tmp)

            tmp = Convolution3D(50, 3, 3, 3, init=initializer, W_regularizer=regularizer)(tmp)
            tmp = PReLU()(tmp)
            tmp = BatchNormalization(axis=1)(tmp)

            tmp = Convolution3D(50, 3, 3, 3, init=initializer, W_regularizer=regularizer)(tmp)
            tmp = PReLU()(tmp)
            tmp = BatchNormalization(axis=1)(tmp)

            tmp = Upsampling3D_mod((downsampling, downsampling, downsampling))(tmp)
            tmp = Convolution3D(50, 3, 3, 3, init=initializer, border_mode='same', W_regularizer=regularizer)(tmp)
            tmp = PReLU()(tmp)
            out_down = BatchNormalization(axis=1)(tmp)

            return out_down

        # Merge two pathways
        concat_layer = Merge(mode='concat', concat_axis=1, output_shape=(100,) + output_shape)(
            [upper_pathway(), lower_pathway()])
        tmp = Convolution3D(150, 1, 1, 1, init=initializer, W_regularizer=regularizer)(concat_layer)
        tmp = PReLU()(tmp)
        tmp = Convolution3D(150, 1, 1, 1, init=initializer, W_regularizer=regularizer)(tmp)
        tmp = PReLU()(tmp)
        tmp = Convolution3D(num_classes, 1, 1, 1, init=initializer, W_regularizer=regularizer)(tmp)
        y = softmax_activation(tmp)

        model = Model(input=x, output=y)
        model.compile(
            optimizer=optimizer,
            loss=categorical_crossentropy_3d,
            metrics=[accuracy, dice_whole, dice_core, dice_enhance, recall_0, recall_1, recall_2, recall_3, recall_4]
        )

        # Load weights if available
        if weights_filename is not None:
            model.load_weights(weights_filename)

        return model, output_shape

    @staticmethod
    def two_pathways_init(num_modalities, segment_dimensions, num_classes, weights_filename=None):
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

        # Hyperaparametre values
        learning_rate = 0.001
        rho = 0.9
        epsilon = 10 ** (-4)
        beta_1 = 0.9
        beta_2 = 0.95

        L1_reg = 0.000001
        L2_reg = 0.0001
        initializer = 'he_normal'

        # Compute input shape, receptive field and output shape after softmax activation
        input_shape = (num_modalities,) + segment_dimensions
        output_shape = segment_dimensions

        # Activations, regularizers and optimizers
        softmax_activation = Lambda(elementwise_softmax_3d, name='softmax')
        regularizer = [L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg)
                       ]
        optimizer = Adam(learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)

        # Model
        x = Input(shape=input_shape, name='Two pathways input')

        # ------- Upper pathway ----
        tmp = Convolution3D(30, 3, 3, 3, init=initializer, border_mode='same', activation='relu',
                            W_regularizer=regularizer[0], name='upper_conv_1')(x)
        tmp = BatchNormalization(axis=1, name='normalization1')(tmp)
        tmp = Convolution3D(30, 3, 3, 3, init=initializer, border_mode='same', activation='relu',
                            W_regularizer=regularizer[1], name='upper_conv_2')(tmp)
        tmp = BatchNormalization(axis=1)(tmp)

        tmp = Convolution3D(40, 3, 3, 3, init=initializer, border_mode='same', activation='relu',
                            W_regularizer=regularizer[2], name='upper_conv_3')(tmp)
        tmp = BatchNormalization(axis=1)(tmp)
        tmp = Convolution3D(40, 3, 3, 3, init=initializer, border_mode='same', activation='relu',
                            W_regularizer=regularizer[3], name='upper_conv_4')(tmp)
        tmp = BatchNormalization(axis=1)(tmp)

        tmp = Convolution3D(40, 3, 3, 3, init=initializer, border_mode='same', activation='relu',
                            W_regularizer=regularizer[4], name='upper_conv_5')(tmp)
        tmp = BatchNormalization(axis=1)(tmp)
        tmp = Convolution3D(40, 3, 3, 3, init=initializer, border_mode='same', activation='relu',
                            W_regularizer=regularizer[5], name='upper_conv_6')(tmp)
        tmp = BatchNormalization(axis=1)(tmp)

        tmp = Convolution3D(50, 3, 3, 3, init=initializer, border_mode='same', activation='relu',
                            W_regularizer=regularizer[6], name='upper_conv_7')(tmp)
        tmp = BatchNormalization(axis=1)(tmp)
        tmp = Convolution3D(50, 3, 3, 3, init=initializer, border_mode='same', activation='relu',
                            W_regularizer=regularizer[7], name='upper_conv_8')(tmp)
        out_up = BatchNormalization(axis=1)(tmp)

        # -------- Lower pathway  ------
        x_down = AveragePooling3D((2, 2, 2))(x)
        tmp = Convolution3D(16, 3, 3, 3, init=initializer, border_mode='same', activation='relu',
                            W_regularizer=regularizer[8], name='lower_conv_1')(x_down)
        tmp = BatchNormalization(axis=1)(tmp)
        tmp = Convolution3D(16, 3, 3, 3, init=initializer, border_mode='same', activation='relu',
                            W_regularizer=regularizer[9], name='lower_conv_2')(tmp)
        tmp = BatchNormalization(axis=1)(tmp)
        tmp = MaxPooling3D((2, 2, 2))(tmp)

        tmp = Convolution3D(32, 3, 3, 3, init=initializer, border_mode='same', activation='relu',
                            W_regularizer=regularizer[10], name='lower_conv_3')(tmp)
        tmp = BatchNormalization(axis=1)(tmp)
        tmp = Convolution3D(32, 3, 3, 3, init=initializer, border_mode='same', activation='relu',
                            W_regularizer=regularizer[11], name='lower_conv_4')(tmp)
        tmp = BatchNormalization(axis=1)(tmp)
        tmp = MaxPooling3D((2, 2, 2))(tmp)

        tmp = Convolution3D(32, 3, 3, 3, init=initializer, border_mode='same', activation='relu',
                            W_regularizer=regularizer[12], name='lower_conv_5')(tmp)
        tmp = BatchNormalization(axis=1)(tmp)
        tmp = Convolution3D(32, 3, 3, 3, init=initializer, border_mode='same', activation='relu',
                            W_regularizer=regularizer[13], name='lower_conv_6')(tmp)
        tmp = BatchNormalization(axis=1)(tmp)
        tmp = MaxPooling3D((2, 2, 2))(tmp)

        tmp = Convolution3D(64, 3, 3, 3, init=initializer, border_mode='same', activation='relu',
                            W_regularizer=regularizer[14], name='lower_conv_7')(tmp)
        tmp = BatchNormalization(axis=1)(tmp)
        tmp = Convolution3D(64, 3, 3, 3, init=initializer, border_mode='same', activation='relu',
                            W_regularizer=regularizer[15], name='lower_conv_8')(tmp)
        tmp = BatchNormalization(axis=1)(tmp)

        tmp = Upsampling3D_mod((2, 2, 2), name='upsampling1')(tmp)
        tmp = Convolution3D(32, 3, 3, 3, init=initializer, border_mode='same', activation='relu',
                            W_regularizer=regularizer[16], name='lower_conv_9')(tmp)
        tmp = BatchNormalization(axis=1)(tmp)
        #
        tmp = Upsampling3D_mod((2, 2, 2), name='upsampling2')(tmp)
        tmp = Convolution3D(32, 3, 3, 3, init=initializer, border_mode='same', activation='relu',
                            W_regularizer=regularizer[17], name='lower_conv_10')(tmp)
        tmp = BatchNormalization(axis=1)(tmp)

        tmp = Upsampling3D_mod((2, 2, 2), name='upsampling3')(tmp)
        tmp = Convolution3D(16, 3, 3, 3, init=initializer, border_mode='same', activation='relu',
                            W_regularizer=regularizer[18], name='lower_conv_11')(tmp)
        tmp = BatchNormalization(axis=1)(tmp)

        tmp = Upsampling3D_mod((2, 2, 2), name='upsampling4')(tmp)
        tmp = Convolution3D(16, 3, 3, 3, init=initializer, border_mode='same', activation='relu',
                            W_regularizer=regularizer[19], name='lower_conv_12')(tmp)
        out_down = BatchNormalization(axis=1)(tmp)

        # Merge two pathways
        concat_layer = Merge(mode='concat', concat_axis=1, output_shape=(100,) + output_shape)(
            [out_up, out_down])
        tmp = Convolution3D(150, 1, 1, 1, init=initializer, activation='relu', W_regularizer=regularizer[20],
                            name='fully_conv_1')(concat_layer)
        tmp = Convolution3D(150, 1, 1, 1, init=initializer, activation='relu', W_regularizer=regularizer[21],
                            name='fully_conv_2')(tmp)
        tmp = Convolution3D(num_classes, 1, 1, 1, init=initializer)(tmp)
        y = softmax_activation(tmp)

        # Create and compile model
        model = Model(input=x, output=y)
        model.compile(
            optimizer=optimizer,
            loss=categorical_crossentropy_3d,
            metrics=[accuracy, dice_whole, dice_core, dice_enhance, recall_0, recall_1, recall_2, recall_3, recall_4,
                     precision_0, precision_1, precision_2, precision_3, precision_4]
        )

        # Load weights if available
        if weights_filename is not None:
            model.load_weights(weights_filename)

        return model, output_shape



    @staticmethod
    def fcn8(num_modalities, segment_dimensions, num_classes, weights_filename=None, pre_train = False):
        """
        Inspired in Long's FCN8 architecture for segmentation using fully convolutional nets and skipped connections
        (http://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf)
            - Convolutional layers: 3x3x3 filters, stride 1, same border
            - Max pooling: 2x2x2
            - Upsampling layers: 2x2x2, 4x4x4, 8x8x8
            - Activations: PreLU (https://arxiv.org/abs/1502.01852)
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

        for dim in segment_dimensions:
            assert dim % 32 == 0  if dim is not None else True # As there are 5 (2, 2, 2) max-poolings, 2 ** 5 is the minimum input size

        # Hyperparametre values
        lr = 0.0001
        beta_1 = 0.9
        beta_2 = 0.999
        epsilon = 10 ** (-6)
        L1_reg = 0.00001
        L2_reg = 0.001
        initializer = 'he_normal'
        pool_size = (2, 2, 2)

        # Compute input shape, receptive field and output shape after softmax activation
        input_shape = (num_modalities,) + segment_dimensions
        output_shape = segment_dimensions

        # Activations, regularizers and optimizers
        softmax_activation = Lambda(elementwise_softmax_3d, name='softmax')
        regularizer = [L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg)
                       ]
        optimizer = Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, clipnorm = 1.)

        # Architecture definition

        # Main pathway
        x = Input(shape=input_shape, name='Long input')

        tmp = Convolution3D(8, 3, 3, 3, init=initializer, border_mode='same', activation='relu',
                            W_regularizer=regularizer[0])(x)
        tmp = BatchNormalization(axis=1)(tmp)
        tmp = Convolution3D(8, 3, 3, 3, init=initializer, border_mode='same', activation='relu',
                            W_regularizer=regularizer[1])(tmp)
        tmp = BatchNormalization(axis=1)(tmp)
        tmp = MaxPooling3D(pool_size=pool_size)(tmp)



        tmp = Convolution3D(16, 3, 3, 3, init=initializer, border_mode='same', activation='relu',
                            W_regularizer=regularizer[2])(tmp)
        tmp = BatchNormalization(axis=1)(tmp)
        tmp = Convolution3D(16, 3, 3, 3, init=initializer, border_mode='same', activation='relu',
                            W_regularizer=regularizer[3])(tmp)
        tmp = BatchNormalization(axis=1)(tmp)
        tmp = MaxPooling3D(pool_size=pool_size)(tmp)



        tmp = Convolution3D(32, 3, 3, 3, init=initializer, border_mode='same', activation='relu',
                            W_regularizer=regularizer[4])(tmp)
        tmp = BatchNormalization(axis=1)(tmp)
        tmp = Convolution3D(32, 3, 3, 3, init=initializer, border_mode='same', activation='relu',
                            W_regularizer=regularizer[5])(tmp)
        tmp = BatchNormalization(axis=1)(tmp)
        tmp = Convolution3D(32, 3, 3, 3, init=initializer, border_mode='same', activation='relu',
                            W_regularizer=regularizer[6])(tmp)
        tmp = BatchNormalization(axis=1)(tmp)
        pool3 = MaxPooling3D(pool_size=pool_size)(tmp)




        tmp = Convolution3D(64, 3, 3, 3, init=initializer, border_mode='same', activation='relu',
                            W_regularizer=regularizer[7])(pool3)
        tmp = BatchNormalization(axis=1)(tmp)
        tmp = Convolution3D(64, 3, 3, 3, init=initializer, border_mode='same', activation='relu',
                            W_regularizer=regularizer[8])(tmp)
        tmp = BatchNormalization(axis=1)(tmp)
        tmp = Convolution3D(64, 3, 3, 3, init=initializer, border_mode='same', activation='relu',
                            W_regularizer=regularizer[9])(tmp)
        tmp = BatchNormalization(axis=1)(tmp)
        pool4 = MaxPooling3D(pool_size=pool_size)(tmp)



        tmp = Convolution3D(64, 3, 3, 3, init=initializer, border_mode='same', activation='relu',
                            W_regularizer=regularizer[10])(pool4)
        tmp = BatchNormalization(axis=1)(tmp)
        tmp = Convolution3D(64, 3, 3, 3, init=initializer, border_mode='same', activation='relu',
                            W_regularizer=regularizer[11])(tmp)
        tmp = BatchNormalization(axis=1)(tmp)
        tmp = Convolution3D(64, 3, 3, 3, init=initializer, border_mode='same', activation='relu',
                            W_regularizer=regularizer[12])(tmp)
        tmp = BatchNormalization(axis=1)(tmp)
        tmp = MaxPooling3D(pool_size=pool_size)(tmp)



        tmp = Convolution3D(512, 1, 1, 1, init=initializer, border_mode='same', activation='relu',
                            W_regularizer=regularizer[13])(tmp)
        tmp = BatchNormalization(axis=1)(tmp)
        tmp = Convolution3D(512, 1, 1, 1, init=initializer, border_mode='same', activation='relu',
                            W_regularizer=regularizer[14])(tmp)
        tmp = BatchNormalization(axis=1)(tmp)
        tmp = Convolution3D(num_classes, 1, 1, 1, init=initializer, border_mode='same',
                            W_regularizer=regularizer[15])(tmp)
        tmp = BatchNormalization(axis=1)(tmp)

        tmp = Upsampling3D_mod(size=(2,2,2))(tmp)
        tmp = Convolution3D(num_classes, 1, 1, 1, init=initializer, border_mode='same',
                            W_regularizer=regularizer[16])(tmp)
        out_pool5 = BatchNormalization(axis=1)(tmp)


        # Skipped layer from pool4
        tmp = Convolution3D(num_classes, 1, 1, 1, init=initializer, border_mode='same',
                            W_regularizer=regularizer[17])(pool4)
        tmp = merge([tmp, out_pool5], mode='sum', name='ActivationsSum_4_5')
        tmp = Upsampling3D_mod(size=(2, 2, 2))(tmp)
        tmp = Convolution3D(num_classes, 1, 1, 1, init=initializer, border_mode='same',
                            W_regularizer=regularizer[18])(tmp)
        out_pool4 = BatchNormalization(axis=1)(tmp)


        # Skipped layer from pool3
        tmp = Convolution3D(num_classes, 1, 1, 1, init=initializer, border_mode='same',
                            W_regularizer=regularizer[19])(pool3)
        tmp = merge([tmp, out_pool4], mode='sum', name='ActivationsSum_3_4_5')

        tmp = Upsampling3D_mod(size=(2, 2, 2))(tmp)
        tmp = Convolution3D(num_classes, 1, 1, 1, init=initializer, border_mode='same',
                            W_regularizer=regularizer[20])(tmp)
        out_pool3 = BatchNormalization(axis=1)(tmp)


        tmp = Upsampling3D_mod(size=(2, 2, 2))(out_pool3)
        tmp = Convolution3D(num_classes, 3, 3, 3, init=initializer, border_mode='same',W_regularizer=regularizer[21])(tmp)
        tmp = BatchNormalization(axis=1)(tmp)

        tmp = Upsampling3D_mod(size=(2, 2, 2))(tmp)
        tmp = Convolution3D(num_classes, 3, 3, 3, init=initializer, border_mode='same',W_regularizer=regularizer[22])(tmp)
        classification = BatchNormalization(axis=1)(tmp)

        y = softmax_activation(classification)


        # Create and compile model

        model = Model(input=x, output=y)
        # if pre_train:
        #     for layer in model.layers[:54]: #51
        #         layer.trainable = False


        model.compile(
            optimizer=optimizer,
            loss=categorical_crossentropy_3d,
            metrics=[accuracy, dice_whole, dice_core, dice_enhance, recall_0, recall_1, recall_2, recall_3, recall_4,
                     precision_0, precision_1, precision_2, precision_3, precision_4]
        )

        # Load weights if available
        if weights_filename is not None:
            model.load_weights(weights_filename)

        return model, output_shape

    @staticmethod
    def fcn8_simple(num_modalities, segment_dimensions, num_classes, weights_filename=None, pre_train = False):
        """
        Inspired in Long's FCN8 architecture for segmentation using fully convolutional nets and skipped connections
        (http://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf)
            - Convolutional layers: 3x3x3 filters, stride 1, same border
            - Max pooling: 2x2x2
            - Upsampling layers: 2x2x2, 4x4x4, 8x8x8
            - Activations: PreLU (https://arxiv.org/abs/1502.01852)
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

        for dim in segment_dimensions:
            assert dim % 32 == 0  if dim is not None else True # As there are 5 (2, 2, 2) max-poolings, 2 ** 5 is the minimum input size

        # Hyperparametre values
        lr = 0.0005
        beta_1 = 0.9
        beta_2 = 0.999
        epsilon = 10 ** (-6)
        L1_reg = 0.00001
        L2_reg = 0.001
        initializer = 'he_normal'
        pool_size = (2, 2, 2)

        # Compute input shape, receptive field and output shape after softmax activation
        input_shape = (num_modalities,) + segment_dimensions
        output_shape = segment_dimensions

        # Activations, regularizers and optimizers
        softmax_activation = Lambda(elementwise_softmax_3d, name='softmax')
        regularizer = [L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg)
                       ]
        optimizer = Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, clipnorm = 1.)

        # Architecture definition

        # Main pathway
        x = Input(shape=input_shape, name='Long input')

        tmp = Convolution3D(8, 3, 3, 3, init=initializer, border_mode='same', activation='relu',
                            W_regularizer=regularizer[0])(x)
        tmp = BatchNormalization(axis=1)(tmp)
        tmp = Convolution3D(8, 3, 3, 3, init=initializer, border_mode='same', activation='relu',
                            W_regularizer=regularizer[1])(tmp)
        tmp = BatchNormalization(axis=1)(tmp)
        tmp = MaxPooling3D(pool_size=pool_size)(tmp)



        tmp = Convolution3D(16, 3, 3, 3, init=initializer, border_mode='same', activation='relu',
                            W_regularizer=regularizer[2])(tmp)
        tmp = BatchNormalization(axis=1)(tmp)
        tmp = Convolution3D(16, 3, 3, 3, init=initializer, border_mode='same', activation='relu',
                            W_regularizer=regularizer[3])(tmp)
        tmp = BatchNormalization(axis=1)(tmp)
        tmp = MaxPooling3D(pool_size=pool_size)(tmp)



        tmp = Convolution3D(32, 3, 3, 3, init=initializer, border_mode='same', activation='relu',
                            W_regularizer=regularizer[4])(tmp)
        tmp = BatchNormalization(axis=1)(tmp)
        tmp = Convolution3D(32, 3, 3, 3, init=initializer, border_mode='same', activation='relu',
                            W_regularizer=regularizer[5])(tmp)
        tmp = BatchNormalization(axis=1)(tmp)
        tmp = Convolution3D(32, 3, 3, 3, init=initializer, border_mode='same', activation='relu',
                            W_regularizer=regularizer[6])(tmp)
        tmp = BatchNormalization(axis=1)(tmp)
        pool3 = MaxPooling3D(pool_size=pool_size)(tmp)




        tmp = Convolution3D(64, 3, 3, 3, init=initializer, border_mode='same', activation='relu',
                            W_regularizer=regularizer[7])(pool3)
        tmp = BatchNormalization(axis=1)(tmp)
        tmp = Convolution3D(64, 3, 3, 3, init=initializer, border_mode='same', activation='relu',
                            W_regularizer=regularizer[8])(tmp)
        tmp = BatchNormalization(axis=1)(tmp)
        tmp = Convolution3D(64, 3, 3, 3, init=initializer, border_mode='same', activation='relu',
                            W_regularizer=regularizer[9])(tmp)
        tmp = BatchNormalization(axis=1)(tmp)
        pool4 = MaxPooling3D(pool_size=pool_size)(tmp)



        tmp = Convolution3D(64, 3, 3, 3, init=initializer, border_mode='same', activation='relu',
                            W_regularizer=regularizer[10])(pool4)
        tmp = BatchNormalization(axis=1)(tmp)
        tmp = Convolution3D(64, 3, 3, 3, init=initializer, border_mode='same', activation='relu',
                            W_regularizer=regularizer[11])(tmp)
        tmp = BatchNormalization(axis=1)(tmp)
        tmp = Convolution3D(64, 3, 3, 3, init=initializer, border_mode='same', activation='relu',
                            W_regularizer=regularizer[12])(tmp)
        tmp = BatchNormalization(axis=1)(tmp)
        tmp = MaxPooling3D(pool_size=pool_size)(tmp)



        tmp = Convolution3D(512, 1, 1, 1, init=initializer, border_mode='same', activation='relu',
                            W_regularizer=regularizer[13])(tmp)
        tmp = BatchNormalization(axis=1)(tmp)
        tmp = Convolution3D(512, 1, 1, 1, init=initializer, border_mode='same', activation='relu',
                            W_regularizer=regularizer[14])(tmp)
        tmp = BatchNormalization(axis=1)(tmp)
        tmp = Convolution3D(num_classes, 1, 1, 1, init=initializer, border_mode='same',
                            W_regularizer=regularizer[15])(tmp)
        tmp = BatchNormalization(axis=1)(tmp)

        tmp = Upsampling3D_mod(size=(2,2,2))(tmp)
        tmp = Convolution3D(num_classes, 3, 3, 3, init=initializer, border_mode='same',
                            W_regularizer=regularizer[16])(tmp)
        out_pool5 = BatchNormalization(axis=1)(tmp)


        # Skipped layer from pool4
        tmp = Upsampling3D_mod(size=(2, 2, 2))(out_pool5)
        tmp = Convolution3D(num_classes, 3, 3, 3, init=initializer, border_mode='same',
                            W_regularizer=regularizer[18])(tmp)
        out_pool4 = BatchNormalization(axis=1)(tmp)


        # Skipped layer from pool3

        tmp = Upsampling3D_mod(size=(2, 2, 2))(out_pool4)
        tmp = Convolution3D(num_classes, 3, 3, 3, init=initializer, border_mode='same',
                            W_regularizer=regularizer[20])(tmp)
        out_pool3 = BatchNormalization(axis=1)(tmp)


        tmp = Upsampling3D_mod(size=(2, 2, 2))(out_pool3)
        tmp = Convolution3D(num_classes, 3, 3, 3, init=initializer, border_mode='same',W_regularizer=regularizer[21])(tmp)
        tmp = BatchNormalization(axis=1)(tmp)

        tmp = Upsampling3D_mod(size=(2, 2, 2))(tmp)
        tmp = Convolution3D(num_classes, 3, 3, 3, init=initializer, border_mode='same',W_regularizer=regularizer[22])(tmp)
        classification = BatchNormalization(axis=1)(tmp)

        y = softmax_activation(classification)


        # Create and compile model

        model = Model(input=x, output=y)
        # if pre_train:
        #     for layer in model.layers[:54]: #51
        #         layer.trainable = False


        model.compile(
            optimizer=optimizer,
            loss=categorical_crossentropy_3d,
            metrics=[accuracy, dice_whole, dice_core, dice_enhance, recall_0, recall_1, recall_2, recall_3, recall_4,
                     precision_0, precision_1, precision_2, precision_3, precision_4]
        )

        # Load weights if available
        if weights_filename is not None:
            model.load_weights(weights_filename)

        return model, output_shape


    @staticmethod
    def u_net(num_modalities, segment_dimensions, num_classes, weights_filename=None, pre_train = False):
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
        lr = 0.0005
        beta_1 = 0.9
        beta_2 = 0.999
        epsilon = 10 ** (-8 )
        L1_reg = 0.000001
        L2_reg = 0.0001
        initializer = 'he_normal'
        pool_size = (2, 2, 2)

        # Compute input shape, receptive field and output shape after softmax activation
        input_shape = (num_modalities,) + segment_dimensions
        output_shape = segment_dimensions

        # Activations, regularizers and optimizers
        softmax_activation = Lambda(elementwise_softmax_3d, name='Softmax')
        regularizer = L1L2(l1=L1_reg, l2=L2_reg)
        optimizer = Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, clipnorm = 1.)

        # Architecture definition

        # First level
        x = Input(shape=input_shape, name='U-net input')
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, name='conv 1.1',
                     activation='relu',
                     padding='same')(x)
        tmp = BatchNormalization(axis=1, name='batch norm 1.1')(tmp)
        tmp = Conv3D(8, (3, 3, 3), kernel_initializer=initializer, name='conv 1.2',  activation='relu',
                     padding='same')(tmp)
        z1 = BatchNormalization(axis=1, name='batch norm 1.2')(tmp)
        tmp = MaxPooling3D(pool_size=pool_size, name='pool 1')(z1)

        # Second level
        tmp = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, name='conv 2.1',  activation='relu',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=1, name='batch norm 2.1')(tmp)
        tmp = Conv3D(16, (3, 3, 3), kernel_initializer=initializer, name='conv 2.2',  activation='relu',
                     padding='same')(tmp)
        z2 = BatchNormalization(axis=1, name='batch norm 2.2')(tmp)
        tmp = MaxPooling3D(pool_size=pool_size, name='pool 2')(z2)

        # Third level
        tmp = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, name='conv 3.1',  activation='relu',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=1, name='batch norm 3.1')(tmp)
        tmp = Conv3D(32, (3, 3, 3), kernel_initializer=initializer, name='conv 3.2',  activation='relu',
                     padding='same')(tmp)
        z3 = BatchNormalization(axis=1, name='batch norm 3.2')(tmp)
        tmp = MaxPooling3D(pool_size=pool_size, name='pool 3')(z3)

        # Fourth level
        tmp = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, name='conv 4.1',  activation='relu',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=1, name='batch norm 4.1')(tmp)
        tmp = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, name='conv 4.2',  activation='relu',
                     padding='same')(tmp)
        z4 = BatchNormalization(axis=1, name='batch norm 4.2')(tmp)
        tmp = MaxPooling3D(pool_size=pool_size, name='pool 4')(z4)

        # Fifth level
        tmp = Conv3D(128, (3, 3, 3), kernel_initializer=initializer, name='conv 5.1',  activation='relu',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=1, name='batch norm 5.1')(tmp)
        tmp = Conv3D(128, (3, 3, 3), kernel_initializer=initializer, name='conv 5.2',  activation='relu',
                     padding='same')(tmp)          #inflection point
        tmp = BatchNormalization(axis=1, name='batch norm 5.2')(tmp)
        tmp = Upsampling3D_mod(size=pool_size, name='up 5')(tmp)

        # Fourth level
        tmp = concatenate([tmp, z4], axis=1, name='merge 4')
        tmp = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, name='conv 4.3', activation='relu',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=1, name='batch norm 4.3')(tmp)
        tmp = Conv3D(64, (3, 3, 3), kernel_initializer=initializer, name='conv 4.4',  activation='relu',
                     padding='same')(tmp)
        tmp = BatchNormalization(axis=1, name='batch norm 4.4')(tmp)
        tmp = Upsampling3D_mod(size=pool_size, name='up 4')(tmp)

        # Third level
        tmp = concatenate([tmp, z3], axis=1, name='merge 3')
        tmp = Convolution3D(32, 3, 3, 3, init=initializer, activation='relu', border_mode='same',
                             name='conv 3.3')(tmp)
        tmp = BatchNormalization(axis=1, name='batch norm 3.3')(tmp)
        tmp = Convolution3D(32, 3, 3, 3, init=initializer, activation='relu', border_mode='same',
                             name='conv 3.4')(tmp)
        tmp = BatchNormalization(axis=1, name='batch norm 3.4')(tmp)
        tmp = Upsampling3D_mod(size=pool_size, name='up 3')(tmp)

        # Second level
        tmp = concatenate([tmp, z2], axis=1, name='merge 2')
        tmp = Convolution3D(16, 3, 3, 3, init=initializer, activation='relu', border_mode='same',
                             name='conv 2.3')(tmp)
        tmp = BatchNormalization(axis=1, name='batch norm 2.3')(tmp)
        tmp = Convolution3D(16, 3, 3, 3, init=initializer, activation='relu', border_mode='same',
                             name='conv 2.4')(tmp)
        tmp = BatchNormalization(axis=1, name='batch norm 2.4')(tmp)
        tmp = Upsampling3D_mod(size=pool_size, name='up 2')(tmp)

        # First level
        tmp = concatenate([tmp, z1], axis=1, name='merge 1')
        tmp = Convolution3D(8, 3, 3, 3, init=initializer, activation='relu', border_mode='same',
                             name='conv 1.3')(tmp)
        tmp = BatchNormalization(axis=1, name='batch norm 1.3')(tmp)
        tmp = Convolution3D(8, 3, 3, 3, init=initializer, activation='relu', border_mode='same',
                             name='conv 1.4')(tmp)
        tmp = BatchNormalization(axis=1, name='batch norm 1.4')(tmp)

        # Classification layer
        classification = Convolution3D(num_classes, 1, 1, 1, activation='relu', init=initializer,
                                       W_regularizer=regularizer, name='final convolution')(tmp)
        classification_norm = BatchNormalization(axis=1, name='final batch norm')(classification)
        y = softmax_activation(classification_norm)

        model = Model(input=x, output=y)
        # if pre_train:
        #     for layer in model.layers[:33]:
        #         layer.trainable = False

        # Create and compile model

        model.compile(
            optimizer=optimizer,
            loss=categorical_crossentropy_3d,
            metrics=[accuracy, dice_whole, dice_core, dice_enhance, recall_0, recall_1, recall_2, recall_3, recall_4,
                     precision_0, precision_1, precision_2, precision_3, precision_4]

        )

        # Load weights if available
        if weights_filename is not None:
            model.load_weights(weights_filename)

        return model, output_shape


    @staticmethod
    def u_net_simple(num_modalities, segment_dimensions, num_classes, weights_filename=None, pre_train = False):
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
        lr = 0.00005
        beta_1 = 0.9
        beta_2 = 0.999
        epsilon = 10 ** (-6 )
        L1_reg = 0.00000
        L2_reg = 0.000
        initializer = 'he_normal'
        pool_size = (2, 2, 2)

        # Compute input shape, receptive field and output shape after softmax activation
        input_shape = (num_modalities,) + segment_dimensions
        output_shape = segment_dimensions

        # Activations, regularizers and optimizers
        softmax_activation = Lambda(elementwise_softmax_3d, name='Softmax')
        optimizer = Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, clipnorm = 1.)
        regularizer = [L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg)
                       ]
        # Architecture definition

        # First level
        x = Input(shape=input_shape, name='U-net input')
        tmp = Convolution3D(8, 3, 3, 3, init=initializer, activation='relu', border_mode='same',
                            W_regularizer = regularizer[0], name='conv 1.1')(x)
        tmp = BatchNormalization(axis=1, name='batch norm 1.1')(tmp)
        tmp = Convolution3D(8, 3, 3, 3, init=initializer, activation='relu', border_mode='same',
                            W_regularizer=regularizer[1], name='conv 1.2')(tmp)
        z1 = BatchNormalization(axis=1, name='batch norm 1.2')(tmp)
        tmp = MaxPooling3D(pool_size=pool_size, name='pool 1')(z1)


        # Second level
        tmp = Convolution3D(16, 3, 3, 3, init=initializer, activation='relu', border_mode='same',
                            W_regularizer=regularizer[2], name='conv 2.1')(tmp)
        tmp = BatchNormalization(axis=1, name='batch norm 2.1')(tmp)
        tmp = Convolution3D(16, 3, 3, 3, init=initializer, activation='relu', border_mode='same',
                            W_regularizer=regularizer[3], name='conv 2.2')(tmp)
        z2 = BatchNormalization(axis=1, name='batch norm 2.2')(tmp)
        tmp = MaxPooling3D(pool_size=pool_size, name='pool 2')(z2)



        # Third level
        tmp = Convolution3D(32, 3, 3, 3, init=initializer, activation='relu', border_mode='same',
                            W_regularizer=regularizer[4], name='conv 3.1')(tmp)
        tmp = BatchNormalization(axis=1, name='batch norm 3.1')(tmp)
        tmp = Convolution3D(32, 3, 3, 3, init=initializer, activation='relu', border_mode='same',
                            W_regularizer=regularizer[5], name='conv 3.2')(tmp)
        z3 = BatchNormalization(axis=1, name='batch norm 3.2')(tmp)
        tmp = MaxPooling3D(pool_size=pool_size, name='pool 3')(z3)

        # Fourth level
        tmp = Convolution3D(64, 3, 3, 3, init=initializer, activation='relu', border_mode='same',
                            W_regularizer=regularizer[6], name='conv 4.1')(tmp)
        tmp = BatchNormalization(axis=1, name='batch norm 4.1')(tmp)
        tmp = Convolution3D(64, 3, 3, 3, init=initializer, activation='relu', border_mode='same',
                            W_regularizer=regularizer[7], name='conv 4.2')(tmp)
        z4 = BatchNormalization(axis=1, name='batch norm 4.2')(tmp)
        tmp = MaxPooling3D(pool_size=pool_size, name='pool 4')(z4)

        # Fifth level
        tmp = Convolution3D(128, 3, 3, 3, init=initializer, activation='relu', border_mode='same',
                            W_regularizer=regularizer[8], name='conv 5.1')(tmp)
        tmp = BatchNormalization(axis=1, name='batch norm 5.1')(tmp)
        tmp = Convolution3D(128, 3, 3, 3, init=initializer, activation='relu', border_mode='same',  # Inflexion point
                            W_regularizer=regularizer[9], name='conv 5.2')(tmp)
        tmp = BatchNormalization(axis=1, name='batch norm 5.2')(tmp)
        tmp = Upsampling3D_mod(size=pool_size, name='up 5')(tmp)
        tmp = Convolution3D(64, 3, 3, 3, init=initializer, border_mode='same', activation='relu',
                            W_regularizer=regularizer[10], name='conv 5.3')(tmp)
        tmp = BatchNormalization(axis=1)(tmp)


        # Fourth level
        tmp = Convolution3D(64, 3, 3, 3, init=initializer, activation='relu', border_mode='same',
                            W_regularizer=regularizer[11], name='conv 4.3')(tmp)
        tmp = BatchNormalization(axis=1, name='batch norm 4.3')(tmp)
        tmp = Convolution3D(64, 3, 3, 3, init=initializer, activation='relu', border_mode='same',
                            W_regularizer=regularizer[12], name='conv 4.4')(tmp)
        tmp = BatchNormalization(axis=1, name='batch norm 4.4')(tmp)
        tmp = Upsampling3D_mod(size=pool_size, name='up 4')(tmp)
        tmp = Convolution3D(32, 3, 3, 3, init=initializer, border_mode='same', activation='relu',
                            W_regularizer=regularizer[13], name='conv 4.5')(tmp)
        tmp = BatchNormalization(axis=1)(tmp)

        # Third level
        tmp = Convolution3D(32, 3, 3, 3, init=initializer, activation='relu', border_mode='same',
                            W_regularizer=regularizer[14], name='conv 3.3')(tmp)
        tmp = BatchNormalization(axis=1, name='batch norm 3.3')(tmp)
        tmp = Convolution3D(32, 3, 3, 3, init=initializer, activation='relu', border_mode='same',
                            W_regularizer=regularizer[15], name='conv 3.4')(tmp)
        tmp = BatchNormalization(axis=1, name='batch norm 3.4')(tmp)
        tmp = Upsampling3D_mod(size=pool_size, name='up 3')(tmp)
        tmp = Convolution3D(16, 3, 3, 3, init=initializer, border_mode='same', activation='relu',
                            W_regularizer=regularizer[16], name='conv 3.5')(tmp)
        tmp = BatchNormalization(axis=1)(tmp)

        # Second level
        tmp = Convolution3D(16, 3, 3, 3, init=initializer, activation='relu', border_mode='same',
                            W_regularizer=regularizer[17], name='conv 2.3')(tmp)
        tmp = BatchNormalization(axis=1, name='batch norm 2.3')(tmp)
        tmp = Convolution3D(16, 3, 3, 3, init=initializer, activation='relu', border_mode='same',
                            W_regularizer=regularizer[18], name='conv 2.4')(tmp)
        tmp = BatchNormalization(axis=1, name='batch norm 2.4')(tmp)
        tmp = Upsampling3D_mod(size=pool_size, name='up 2')(tmp)
        tmp = Convolution3D(8, 3, 3, 3, init=initializer, border_mode='same', activation='relu',
                            W_regularizer=regularizer[19], name='conv 2.5')(tmp)
        tmp = BatchNormalization(axis=1)(tmp)

        # First level
        tmp = Convolution3D(8, 3, 3, 3, init=initializer, activation='relu', border_mode='same',
                            W_regularizer=regularizer[20], name='conv 1.3')(tmp)
        tmp = BatchNormalization(axis=1, name='batch norm 1.3')(tmp)
        tmp = Convolution3D(8, 3, 3, 3, init=initializer, activation='relu', border_mode='same',
                            W_regularizer=regularizer[21], name='conv 1.4')(tmp)
        tmp = BatchNormalization(axis=1, name='batch norm 1.4')(tmp)

        # Classification layer
        classification = Convolution3D(num_classes, 1, 1, 1, activation='relu', init=initializer,name='final convolution')(tmp)
        y = softmax_activation(classification)

        model = Model(input=x, output=y)
        # if pre_train:
        #     for layer in model.layers[:33]:
        #         layer.trainable = False

        # Create and compile model

        model.compile(
            optimizer=optimizer,
            loss=categorical_crossentropy_3d,
            metrics=[accuracy, dice_whole, dice_core, dice_enhance, recall_0, recall_1, recall_2, recall_3, recall_4,
                     precision_0, precision_1, precision_2, precision_3, precision_4]

        )

        # Load weights if available
        if weights_filename is not None:
            model.load_weights(weights_filename)

        return model, output_shape


    @staticmethod
    def two_pathways(num_modalities, segment_dimensions, num_classes, weights_filename=None, pre_train = False):
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

        # Hyperaparametre values
        learning_rate = 0.001
        rho = 0.9
        epsilon = 10 ** (-4)
        beta_1 = 0.9
        beta_2 = 0.95

        L1_reg = 0.000001
        L2_reg = 0.0001
        initializer = 'he_normal'

        # Compute input shape, receptive field and output shape after softmax activation
        input_shape = (num_modalities,) + segment_dimensions
        output_shape = segment_dimensions

        # Activations, regularizers and optimizers
        softmax_activation = Lambda(elementwise_softmax_3d, name='softmax')
        regularizer = [L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg)
                       ]
        optimizer = Adam(learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, clipnorm=1.)

        # Model
        x = Input(shape=input_shape, name='Two pathways input')

        # ------- Upper pathway ----
        tmp = Convolution3D(8, 3, 3, 3, init=initializer, border_mode='same', activation='relu',
                            W_regularizer=regularizer[0], name='upper_conv_1')(x)
        tmp = BatchNormalization(axis=1, name='normalization1')(tmp)
        tmp = Convolution3D(8, 3, 3, 3, init=initializer, border_mode='same', activation='relu',
                            W_regularizer=regularizer[1], name='upper_conv_2')(tmp)
        tmp = BatchNormalization(axis=1)(tmp)

        tmp = Convolution3D(8, 3, 3, 3, init=initializer, border_mode='same', activation='relu',
                            W_regularizer=regularizer[2], name='upper_conv_3')(tmp)
        tmp = BatchNormalization(axis=1)(tmp)
        tmp = Convolution3D(8, 3, 3, 3, init=initializer, border_mode='same', activation='relu',
                            W_regularizer=regularizer[3], name='upper_conv_4')(tmp)
        tmp = BatchNormalization(axis=1)(tmp)

        tmp = Convolution3D(8, 3, 3, 3, init=initializer, border_mode='same', activation='relu',
                            W_regularizer=regularizer[4], name='upper_conv_5')(tmp)
        tmp = BatchNormalization(axis=1)(tmp)
        tmp = Convolution3D(8, 3, 3, 3, init=initializer, border_mode='same', activation='relu',
                            W_regularizer=regularizer[5], name='upper_conv_6')(tmp)
        tmp = BatchNormalization(axis=1)(tmp)

        tmp = Convolution3D(8, 3, 3, 3, init=initializer, border_mode='same', activation='relu',
                            W_regularizer=regularizer[6], name='upper_conv_7')(tmp)
        tmp = BatchNormalization(axis=1)(tmp)
        tmp = Convolution3D(8, 3, 3, 3, init=initializer, border_mode='same', activation='relu',
                            W_regularizer=regularizer[7], name='upper_conv_8')(tmp)
        out_up = BatchNormalization(axis=1)(tmp)

        # -------- Lower pathway  ------
        x_down = AveragePooling3D((2, 2, 2))(x)
        tmp = Convolution3D(16, 3, 3, 3, init=initializer, border_mode='same', activation='relu',
                            W_regularizer=regularizer[8], name='lower_conv_1')(x_down)
        tmp = BatchNormalization(axis=1)(tmp)
        tmp = Convolution3D(16, 3, 3, 3, init=initializer, border_mode='same', activation='relu',
                            W_regularizer=regularizer[9], name='lower_conv_2')(tmp)
        tmp = BatchNormalization(axis=1)(tmp)
        tmp = MaxPooling3D((2, 2, 2))(tmp)

        tmp = Convolution3D(32, 3, 3, 3, init=initializer, border_mode='same', activation='relu',
                            W_regularizer=regularizer[10], name='lower_conv_3')(tmp)
        tmp = BatchNormalization(axis=1)(tmp)
        tmp = Convolution3D(32, 3, 3, 3, init=initializer, border_mode='same', activation='relu',
                            W_regularizer=regularizer[11], name='lower_conv_4')(tmp)
        tmp = BatchNormalization(axis=1)(tmp)
        tmp = MaxPooling3D((2, 2, 2))(tmp)

        tmp = Convolution3D(32, 3, 3, 3, init=initializer, border_mode='same', activation='relu',
                            W_regularizer=regularizer[12], name='lower_conv_5')(tmp)
        tmp = BatchNormalization(axis=1)(tmp)
        tmp = Convolution3D(32, 3, 3, 3, init=initializer, border_mode='same', activation='relu',
                            W_regularizer=regularizer[13], name='lower_conv_6')(tmp)
        tmp = BatchNormalization(axis=1)(tmp)
        tmp = MaxPooling3D((2, 2, 2))(tmp)

        tmp = Convolution3D(64, 3, 3, 3, init=initializer, border_mode='same', activation='relu',
                            W_regularizer=regularizer[14], name='lower_conv_7')(tmp)
        tmp = BatchNormalization(axis=1)(tmp)
        tmp = Convolution3D(64, 3, 3, 3, init=initializer, border_mode='same', activation='relu',
                            W_regularizer=regularizer[15], name='lower_conv_8')(tmp)
        tmp = BatchNormalization(axis=1)(tmp)

        tmp = Upsampling3D_mod((2, 2, 2), name='upsampling1')(tmp)
        tmp = Convolution3D(32, 3, 3, 3, init=initializer, border_mode='same', activation='relu',
                            W_regularizer=regularizer[16], name='lower_conv_9')(tmp)
        tmp = BatchNormalization(axis=1)(tmp)
        #
        tmp = Upsampling3D_mod((2, 2, 2), name='upsampling2')(tmp)
        tmp = Convolution3D(32, 3, 3, 3, init=initializer, border_mode='same', activation='relu',
                            W_regularizer=regularizer[17], name='lower_conv_10')(tmp)
        tmp = BatchNormalization(axis=1)(tmp)

        tmp = Upsampling3D_mod((2, 2, 2), name='upsampling3')(tmp)
        tmp = Convolution3D(16, 3, 3, 3, init=initializer, border_mode='same', activation='relu',
                            W_regularizer=regularizer[18], name='lower_conv_11')(tmp)
        tmp = BatchNormalization(axis=1)(tmp)

        tmp = Upsampling3D_mod((2, 2, 2), name='upsampling4')(tmp)
        tmp = Convolution3D(8, 3, 3, 3, init=initializer, border_mode='same', activation='relu',
                            W_regularizer=regularizer[19], name='lower_conv_12')(tmp)
        out_down = BatchNormalization(axis=1)(tmp)

        # Merge two pathways
        concat_layer = Merge(mode='concat', concat_axis=1, output_shape=(100,) + output_shape)(
            [out_up, out_down])
        tmp = Convolution3D(16, 1, 1, 1, init=initializer, activation='relu', W_regularizer=regularizer[20],
                            name='fully_conv_1')(concat_layer)
        tmp = Convolution3D(16, 1, 1, 1, init=initializer, activation='relu', W_regularizer=regularizer[21],
                            name='fully_conv_2')(tmp)
        tmp = Convolution3D(num_classes, 1, 1, 1, init=initializer)(tmp)
        y = softmax_activation(tmp)

        # Create and compile model
        model = Model(input=x, output=y)
        # if pre_train:
        #     for layer in model.layers[:50]:
        #         layer.trainable = False

        model.compile(
            optimizer=optimizer,
            loss=categorical_crossentropy_3d,
            metrics=[accuracy, dice_whole, dice_core, dice_enhance, recall_0, recall_1, recall_2, recall_3,
                     recall_4,
                     precision_0, precision_1, precision_2, precision_3, precision_4]
        )

        # Load weights if available
        if weights_filename is not None:
            model.load_weights(weights_filename)

        return model, output_shape

    @staticmethod
    def two_pathways_simple(num_modalities, segment_dimensions, num_classes, weights_filename=None, pre_train = False):
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

        # Hyperaparametre values
        learning_rate = 0.001
        rho = 0.9
        epsilon = 10 ** (-4)
        beta_1 = 0.9
        beta_2 = 0.95

        L1_reg = 0.000001
        L2_reg = 0.0001
        initializer = 'he_normal'

        # Compute input shape, receptive field and output shape after softmax activation
        input_shape = (num_modalities,) + segment_dimensions
        output_shape = segment_dimensions

        # Activations, regularizers and optimizers
        softmax_activation = Lambda(elementwise_softmax_3d, name='softmax')
        regularizer = [L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg),
                       L1L2(l1=L1_reg, l2=L2_reg)
                       ]
        optimizer = Adam(learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, clipnorm=1.)

        # Model
        x = Input(shape=input_shape, name='Two pathways input')

        # ------- Upper pathway ----
        tmp = Convolution3D(8, 3, 3, 3, init=initializer, border_mode='same', activation='relu',
                            W_regularizer=regularizer[0], name='upper_conv_1')(x)
        tmp = BatchNormalization(axis=1, name='normalization1')(tmp)
        tmp = Convolution3D(8, 3, 3, 3, init=initializer, border_mode='same', activation='relu',
                            W_regularizer=regularizer[1], name='upper_conv_2')(tmp)
        tmp = BatchNormalization(axis=1)(tmp)

        tmp = Convolution3D(8, 3, 3, 3, init=initializer, border_mode='same', activation='relu',
                            W_regularizer=regularizer[2], name='upper_conv_3')(tmp)
        tmp = BatchNormalization(axis=1)(tmp)
        tmp = Convolution3D(8, 3, 3, 3, init=initializer, border_mode='same', activation='relu',
                            W_regularizer=regularizer[3], name='upper_conv_4')(tmp)
        tmp = BatchNormalization(axis=1)(tmp)

        tmp = Convolution3D(8, 3, 3, 3, init=initializer, border_mode='same', activation='relu',
                            W_regularizer=regularizer[4], name='upper_conv_5')(tmp)
        tmp = BatchNormalization(axis=1)(tmp)
        tmp = Convolution3D(8, 3, 3, 3, init=initializer, border_mode='same', activation='relu',
                            W_regularizer=regularizer[5], name='upper_conv_6')(tmp)
        tmp = BatchNormalization(axis=1)(tmp)

        tmp = Convolution3D(8, 3, 3, 3, init=initializer, border_mode='same', activation='relu',
                            W_regularizer=regularizer[6], name='upper_conv_7')(tmp)
        tmp = BatchNormalization(axis=1)(tmp)
        tmp = Convolution3D(8, 3, 3, 3, init=initializer, border_mode='same', activation='relu',
                            W_regularizer=regularizer[7], name='upper_conv_8')(tmp)
        out_up = BatchNormalization(axis=1)(tmp)

        # Merge two pathways
        tmp = Convolution3D(16, 1, 1, 1, init=initializer, activation='relu', W_regularizer=regularizer[20],
                            name='fully_conv_1')(out_up)
        tmp = Convolution3D(16, 1, 1, 1, init=initializer, activation='relu', W_regularizer=regularizer[21],
                            name='fully_conv_2')(tmp)
        tmp = Convolution3D(num_classes, 1, 1, 1, init=initializer)(tmp)
        y = softmax_activation(tmp)

        # Create and compile model
        model = Model(input=x, output=y)
        if pre_train:
            for layer in model.layers[:50]:
                layer.trainable = False

        model.compile(
            optimizer=optimizer,
            loss=categorical_crossentropy_3d,
            metrics=[accuracy, dice_whole, dice_core, dice_enhance, recall_0, recall_1, recall_2, recall_3,
                     recall_4,
                     precision_0, precision_1, precision_2, precision_3, precision_4]
        )

        # Load weights if available
        if weights_filename is not None:
            model.load_weights(weights_filename)

        return model, output_shape

