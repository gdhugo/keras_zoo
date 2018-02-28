#!/usr/bin/env python
import argparse
import os
import sys
from getpass import getuser
import matplotlib
matplotlib.use('Agg')  # Faster plot

# Import tools
from config.configuration import Configuration
from tools.logger import Logger
from tools.dataset_generators import Dataset_Generators
from tools.optimizer_factory import Optimizer_Factory
from callbacks.callbacks_factory import Callbacks_Factory
from models.model_factory import Model_Factory


# Train the network
def process(configuration):
    # Load configuration
    cf = configuration.load()

    # Enable log file
    sys.stdout = Logger(cf.log_file)
    print (' ---> Init experiment: ' + cf.exp_name + ' <---')

    # Create the data generators
    if cf.dataset.data_format != 'npz':
        train_gen, valid_gen, test_gen = Dataset_Generators().make(cf)
    else:
        train_gen, valid_gen, test_gen = None, None, None

    # Create the optimizer
    print ('\n > Creating optimizer...')
    optimizer = Optimizer_Factory().make(cf)

    # Build model
    print ('\n > Building model...')
    model = Model_Factory().make(cf, optimizer)

    # Create the callbacks
    print ('\n > Creating callbacks...')
    cb = Callbacks_Factory().make(cf, valid_gen)

    try:
        if cf.train_model:
            # Train the model
            if cf.dataset.data_format == 'npz':
                model.train2(cf.dataset.load_data_func, cb)
            elif cf.dataset.data_format == 'folders':
                model.train(train_gen, valid_gen, cb)
            else:
                raise ValueError('Unknown data format')

        if cf.test_model:
            # Compute validation metrics
            model.test(valid_gen)
            # Compute test metrics
            model.test(test_gen)

        if cf.pred_model:
            # Compute validation metrics
            model.predict(valid_gen, tag='pred')
            # Compute test metrics
            model.predict(test_gen, tag='pred')

    except KeyboardInterrupt:
        # In case of early stopping, transfer the local files
        do_copy = raw_input('\033[93m KeyboardInterrupt \nDo you want to transfer files to {} ? ([y]/n) \033[0m'
                            .format(cf.final_savepath))
        if do_copy in ['', 'y']:
            # Copy result to shared directory
            configuration.copy_to_shared()
        raise

    # Finish
    print (' ---> Finish experiment: ' + cf.exp_name + ' <---')


# Sets the backend and GPU device.
class Environment():
    def __init__(self, backend='tensorflow'):
        #backend = 'tensorflow' # 'theano' or 'tensorflow'
        os.environ['KERAS_BACKEND'] = backend
        os.environ["CUDA_VISIBLE_DEVICES"]="0" # "" to run in CPU, extra slow! just for debuging
        if backend == 'theano':
            # os.environ['THEANO_FLAGS']='mode=FAST_RUN,device=gpu1,floatX=float32,optimizer=fast_compile'
            """ fast_compile que lo que hace es desactivar las optimizaciones => mas lento """
            os.environ['THEANO_FLAGS'] = 'device=gpu0,floatX=float32,lib.cnmem=0.95'
            print('Backend is Theano now')
        else:
            print('Backend is Tensorflow now')


# Main function
def main():
    # Define environment variables
    # Environment()

    # Get parameters from arguments
    parser = argparse.ArgumentParser(description='Model training')
    parser.add_argument('-c', '--config_path', type=str,
                        default=None, help='Configuration file')
    parser.add_argument('-e', '--exp_name', type=str,
                        default=None, help='Name of the experiment')
    parser.add_argument('-s', '--shared_path', type=str,
                        default='/data', help='Shared path')
    parser.add_argument('-l', '--local_path', type=str,
                        default='/datatmp', help='Local path')

    arguments = parser.parse_args()

    assert arguments.config_path is not None, 'Please provide a configuration'\
                                              'path using -c config/pathname'\
                                              ' in the command line'
    assert arguments.exp_name is not None, 'Please provide a name for the '\
                                           'experiment using -e name in the '\
                                           'command line'

    # Define the user paths
    shared_path = arguments.shared_path
    local_path = arguments.local_path
    dataset_path = os.path.join(local_path, 'Datasets')
    shared_dataset_path = os.path.join(shared_path, 'Datasets')
    experiments_path = os.path.join(local_path, getuser(), 'Experiments')
    shared_experiments_path = os.path.join(shared_path, getuser(), 'Experiments')
    usr_path = os.path.join(os.path.expanduser("~"), getuser())

    # Load configuration files
    configuration = Configuration(arguments.config_path, arguments.exp_name,
                                  dataset_path, shared_dataset_path,
                                  experiments_path, shared_experiments_path,
                                  usr_path)

    # Train /test/predict with the network, depending on the configuration
    process(configuration)

    # Copy result to shared directory
    configuration.copy_to_shared()


# Entry point of the script
if __name__ == "__main__":
    main()
