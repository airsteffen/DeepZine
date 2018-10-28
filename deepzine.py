
import os
import tables
import numpy as np
import math

from download_internet_archive import internet_archive_download, convert_pdf_to_image, store_to_hdf5, PageData
from utils import add_parameter
from model import PGGAN


class DeepZine(object):

    def __init__(self, **kwargs):

        # General Parameters
        add_parameter(self, kwargs, 'load_data', True)
        add_parameter(self, kwargs, 'train', True)
        add_parameter(self, kwargs, 'inference', True)

        # Model Parameters -- more important for test/reverse, maybe
        add_parameter(self, kwargs, 'progressive_depth', 4)
        add_parameter(self, kwargs, 'starting_depth', 0)

        # Train Data Parameters
        add_parameter(self, kwargs, 'data_hdf5', None)
        add_parameter(self, kwargs, 'pdf_directory', None)
        add_parameter(self, kwargs, 'image_directory', None)
        add_parameter(self, kwargs, 'overwrite', False)

        add_parameter(self, kwargs, 'download_pdf', True)
        add_parameter(self, kwargs, 'internetarchive_collection', 'MBLWHOI')
        add_parameter(self, kwargs, 'convert_pdf', True)
        
        add_parameter(self, kwargs, 'pdf_num', None)  # None == whole collection.
        add_parameter(self, kwargs, 'data_output_size', 1024)
        add_parameter(self, kwargs, 'preload_resized_data', True)

        # Training GAN Parameters
        add_parameter(self, kwargs, 'samples_dir', './samples')
        add_parameter(self, kwargs, 'log_dir', './log')
        add_parameter(self, kwargs, 'progressive_depth', None)
        add_parameter(self, kwargs, 'gan_output_size', 128)

        # Inference Parameters
        add_parameter(self, kwargs, 'test_data_directory', None)
        add_parameter(self, kwargs, 'test_model_path', None)
        add_parameter(self, kwargs, 'test_model_samples', 100)
        add_parameter(self, kwargs, 'test_input_latent', 100)

        # Derived Parameters
        if self.progressive_depth is None:
            self.progressive_depth = int(math.log(self.gan_output_size, 2) - 1)
        self.training_storage = None

        self.kwargs = kwargs

        return

    def execute(self):

        if self.train or self.load_data:

            # Data preparation.
            self.training_storage = self.download_data()

            if self.train:
            
                try:
                    self.train_gan()
                except:
                    self.close_storage()
                    raise

                self.close_storage()

        if self.inference:

            self.test_gan()

        self.close_storage()

        return

    def close_storage(self):
        if self.training_storage is not None:
            self.training_storage.close()
            self.training_storage = None

    def download_data(self):

        # Check if an HDF5 exists, otherwise initiate the process of creating one.
        if self.data_hdf5 is None:
            raise ValueError('Please provide an HDF5 file to stream data from.')
        else:
            if os.path.exists(self.data_hdf5) and not self.overwrite:
                output_hdf5 = self.data_hdf5
            else:
                output_hdf5 = None

        if output_hdf5 is None:

            # Create a working data_directory if necessary.
            if not os.path.exists(self.pdf_directory) and not self.download_pdf:
                raise ValueError('Data directory not found.')
            elif not os.path.exists(self.pdf_directory):
                os.mkdir(self.pdf_directory)

            # Download data
            if self.download_pdf:
                internet_archive_download(self.pdf_directory, self.internetarchive_collection, self.pdf_num)

            # Convert PDFs into images.
            if self.convert_pdf:
                if not os.path.exists(self.image_directory):
                    os.mkdir(self.image_directory)
                convert_pdf_to_image(self.pdf_directory, self.image_directory)

            # Preprocess images and write to HDF5.
            output_hdf5 = store_to_hdf5(self.image_directory, self.data_hdf5, self.data_output_size)

        output_hdf5 = tables.open_file(output_hdf5, "r")

        # Convert to data-loading object. The logic is all messed up here for pre-loading images.
        return PageData(hdf5=output_hdf5, output_size=self.gan_output_size)

    def train_gan(self):

        # Create necessary directories
        for work_dir in [self.samples_dir, self.log_dir]:
            if not os.path.exists(work_dir):
                os.mkdir(work_dir)

        # Some explanation on training stages: The progressive gan trains each resolution
        # in two stages. One interpolates from the previous resolution, while one trains 
        # solely on the current resolution. The loop below looks odd because the lowest 
        # resolution only has one stage.

        print(self.starting_depth, self.progressive_depth)
        print((self.progressive_depth * 2) - 1)
        print(int(np.ceil((self.starting_depth - 1) / 2)))
        print(range(int(np.ceil((self.starting_depth - 1) / 2)), (self.progressive_depth * 2) - 1))
        for training_stage in range(int(np.ceil((self.starting_depth - 1) / 2)), (self.progressive_depth * 2) - 1):

            if (training_stage % 2 == 0):
                transition = False
                transition_string = ''
            else:
                transition = True
                transition_string = '_Transition'

            current_depth = np.ceil((training_stage + 1) / 2)
            previous_depth = np.ceil((training_stage) / 2)

            current_size = int(2 * 2 ** current_depth)
            previous_size = int(2 * 2 ** previous_depth)

            print(current_size, previous_size)
            print(current_depth, previous_depth)

            output_model_path = os.path.join(self.log_dir, str(current_size), 'model.ckpt')
            if not os.path.exists(os.path.dirname(output_model_path)):
                os.mkdir(os.path.dirname(output_model_path))

            input_model_path = os.path.join(self.log_dir, str(previous_size), 'model.ckpt')

            sample_path = os.path.join(self.samples_dir, 'samples_' + str(current_size) + transition_string)
            if not os.path.exists(sample_path):
                os.mkdir(sample_path)

            print(input_model_path, output_model_path, sample_path)

            pggan = PGGAN(training_data=self.training_storage,
                            input_model_path=input_model_path, 
                            output_model_path=output_model_path,
                            model_sample_dir=sample_path, 
                            model_logging_dir=self.log_dir,
                            model_output_size=current_size,
                            transition=transition,
                            **self.kwargs)

            pggan.build_model()
            pggan.train()

    def test_gan(self, input_latent=None):

        if not os.path.exists(self.test_data_directory):
            os.makedirs(self.test_data_directory)

        if self.test_input_latent is None:
            pggan = PGGAN(input_model_path=self.test_model_path,
                            progressive_depth=self.progressive_depth,
                            testing=True,
                            channel=self.channels)

            pggan.build_model()
            pggan.test_model(self.test_data_directory, self.test_model_samples)      

        else:
            for i in xrange(1, 9):
                pggan = PGGAN(input_model_path=os.path.join(self.test_model_path, str(i), 'model.ckpt'),
                                progressive_depth=i,
                                testing=True,
                                channels=self.channels)

                pggan.build_model()
                pggan.test_model(self.test_data_directory, self.test_model_samples, input_latent=self.test_input_latent)      


if __name__ == '__main__':

    pass