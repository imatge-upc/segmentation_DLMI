import numpy as np
import time
import keras.backend as K


def get_activations(model, layer_name,input_img):
    layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
    get_activations = K.function([model.layers[0].input, K.learning_phase()], layer_dict[layer_name].output)
    activations = get_activations([input_img,0])
    return activations



def propagate_filters(input_img_data,model,layer_name,image_size,n_filters=-1,n_steps = 20):
    def normalize(x):
        # utility function to normalize a tensor by its L2 norm
        return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

    def deprocess_image(x):
        # normalize tensor: center on 0., ensure std is 0.1
        x -= x.mean()
        x /= (x.std() + 1e-5)
        x *= 0.1

        return x

    input_img = model.input
    layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
    layer_output = layer_dict[layer_name].output



    kept_filters = []
    for filter_index in range(0, n_filters):
        print('Processing filter %d' % filter_index)
        start_time = time.time()

        # We build a loss function that maximizes the activation
        # of the nth filter of the layer considered

        if K.image_dim_ordering() == 'th':
            loss = K.mean(layer_output[:, filter_index, :, :, :])
        else:
            loss = K.mean(layer_output[:, :, :, :, filter_index])

        # We compute the gradient this loss wrt the image input
        grads = K.gradients(loss, input_img)[0]

        # normalization trick: we normalize the gradient
        grads = normalize(grads)

        # this function returns the loss and grads given the input picture
        iterate = K.function([input_img], [loss, grads])

        # step size for gradient ascent
        step = 500.

        # we start from a gray image with some random noise
        # if K.image_dim_ordering() == 'th':
        #     input_img_data = np.random.random((1, 4, image_size[0],image_size[1],image_size[2]))
        # else:
        #     input_img_data = np.random.random((1, image_size[0],image_size[1],image_size[2],4))
        # input_img_data= (input_img_data - 0.5) * 20 + 128
        #

        # we run gradient ascent for N steps
        for i in range(n_steps):
            loss_value, grads_value = iterate([input_img_data])
            input_img_data += grads_value * step

            print('Current loss value:', loss_value)
            if loss_value <= 0.000001 or np.isnan(loss_value):
                # some filters get stuck to 0, we can skip them
                print('These filter got stuck to 0. Skip it')
                break

        # decode the resulting input image
        if loss_value > 0:
            img = deprocess_image(input_img_data[0])
            kept_filters.append(img)
        end_time = time.time()
        print('Filter %d processed in %ds' % (filter_index, end_time - start_time))

    return kept_filters
