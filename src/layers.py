from keras import backend as K
from keras.layers import Lambda, Convolution3D, UpSampling3D



class Upsampling3D_mod(UpSampling3D):
    def get_output_shape_for(self, input_shape):

        if self.dim_ordering == 'th':
            conv_dim1 = input_shape[2] * self.size[0] if input_shape[2] is not None else None
            conv_dim2 = input_shape[3] * self.size[1] if input_shape[3] is not None else None
            conv_dim3 = input_shape[4] * self.size[2] if input_shape[4] is not None else None
        elif self.dim_ordering == 'tf':
            conv_dim1 = input_shape[1] * self.size[0] if input_shape[1] is not None else None
            conv_dim2 = input_shape[2] * self.size[0] if input_shape[2] is not None else None
            conv_dim3 = input_shape[3] * self.size[0] if input_shape[3] is not None else None
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

        if self.dim_ordering == 'th':
            return (input_shape[0], input_shape[1], conv_dim1, conv_dim2, conv_dim3)
        elif self.dim_ordering == 'tf':
            return (input_shape[0], conv_dim1, conv_dim2, conv_dim3, input_shape[4])
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

