from __future__ import print_function
import os
from argparse import ArgumentParser
from keras.utils import visualize_util
from src.models import BratsModels

if __name__ == '__main__':

    model_choices = BratsModels.get_model_names()
    model_choices.append('all')
    default_model = BratsModels.DEFAULT_MODEL

    arg_parser = ArgumentParser(description='Saves a model representation in PNG format in the specified folder.')

    arg_parser.add_argument('model', choices=model_choices, default=default_model,
                            help='Name of the model to represent.')

    arg_parser.add_argument('output_dir', help='Directory into which the png file representing the model should be '
                                               'saved')

    args = arg_parser.parse_args()
    model_name = args.model
    output_dir = args.output_dir

    # Check if output dir exists
    if not os.path.isdir(output_dir):
        print('The specified output directory does not exist.')
        exit(1)

    if model_name != 'all':
        # Output file name
        filename = os.path.join(output_dir, '{}.png'.format(model_name))

        # Get model
        model, output_shape = BratsModels.get_model(4, (64, 64, 64), 5, model_name=model_name)

        # Represent model
        visualize_util.plot(model, to_file=filename, show_shapes=True)
    else:
        # Pop 'all' choice
        model_choices.pop()
        for model_n in model_choices:
            # Output file name
            filename = os.path.join(output_dir, '{}.png'.format(model_n))
            try:
                # Get model
                model, output_shape = BratsModels.get_model(4, (64, 64, 64), 5, model_name=model_n)
                # Represent model
                visualize_util.plot(model, to_file=filename, show_shapes=True)
                print('{} done'.format(model_n))
            except ValueError:
                print('Model {} was skipped due to an error.'.format(model_n))
