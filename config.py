"""

"""


import os
import numpy as np
import argparse
import sys


class DeepZine_cli(object):

    def __init__(self):

        parser = argparse.ArgumentParser(
            description='Some commands for generating synthetic book pages with a PGGAN.',
            usage='''deepzine <command> [<args>]
                    The following commands are available:
                        load_data              Use the internetarchive API to download data.
                        train_model            Train a model based on an hdf5 created in load_data.
                        run_inference          Run inference on a trained model. 
                ''')

        parser.add_argument('command', help='Subcommand to run')
        args = parser.parse_args(sys.argv[1:2])

        if not hasattr(self, args.command):
            print('Sorry, that\'s not one of the commands.')
            parser.print_help()
            exit(1)

        getattr(self, args.command)()

    def load_data(self):

        parser = argparse.ArgumentParser(
            description='''deepzine load_data
            Stream data from the internetarchive into an hdf5 file, which can then be used for training a neural network.

            -output_pdf_folder:
            -output_images_folder:
            -output_hdf5:

            -maximum_output_size: Maximum output size for image generation. Must be a power of 2.
            -number_of_pdfs: Maximum number of pdfs to download.
            -internetarchive_collection: Code for the collection you wish download from on the Internet Archive.
            -overwrite: If the target file already exists, overwrite it.
                ''')

        parser.add_argument('-output_pdf_folder', type=str)
        parser.add_argument('-output_images_folder', type=str)
        parser.add_argument('-output_hdf5', type=str)

        parser.add_argument('-maximum_output_size', type=int, default=1024)
        parser.add_argument('-number_of_pdfs', type=int, default=10000)
        parser.add_argument('-internetarchive_collection', type=str, const='MBLWHOI', default='MBLWHOI')
        parser.add_argument('-overwrite', action='store_true') 

        args = parser.parse_args(sys.argv[2:])

        from deepzine import DeepZine

        gan = DeepZine(load_data=True,
                        gan_output_size=args.maximum_output_size,
                        pdf_num=args.number_of_pdfs,
                        download_pdf=True,
                        convert_pdf=True,
                        overwrite=args.overwrite,
                        collection=args.internetarchive_collection,
                        pdf_directory=args.output_pdf_folder,
                        image_directory=args.output_images_folder,
                        data_hdf5=args.output_hdf5)

        gan.execute()

    def train_model(self):

        parser = argparse.ArgumentParser(
            description='''deepzine train_model
            Stream data from the internetarchive into an hdf5 file, which can then be used for training a neural network.

            -samples_folder: At given intervals, samples from the GAN will be output here.
            -log_folder: Models and tensorboard logs will be written to this folder.

            -output_size: Final training size of the PGGAN. Must be a power of 2.
            -gpu_num: If you have multiple GPUs and only want to use one. Numbering start at 0.
                ''')

        parser.add_argument('-samples_folder', type=str)
        parser.add_argument('-log_folder', type=str)
        parser.add_argument('-output_size', type=int, default=1024)
        parser.add_argument('-gpu_num', nargs='?', const='0', default='0', type=str)

        args = parser.parse_args(sys.argv[2:])    

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
        from deepzine import DeepZine

        """ Note: there's a lot of other parameters you can pass in here
            to change the hyperparameters of the PGGAN model. They are
            passed through the DeepZine object on to the PGGAN object
            in model.py.
        """

        gan = DeepZine(train=True,
                        gan_output_size=args.output_size,
                        samples_dir=args.samples_folder,
                        log_dir=args.log_folder)

        gan.execute()

    def run_inference(self):

        parser = argparse.ArgumentParser(
            description='''deepzine run_inference
            Run inference on a trained model.

            -model_folder: Directory that contains models at each trained resoultion produced by previous step.
            -output_folder: Directory where images will be created.

            -output_num: Number of images to create on inference.
            -output_size: Final training size of the PGGAN. Must be a power of 2.
            -gpu_num: If you have multiple GPUs and only want to use one. Numbering start at 0.
                ''')

        parser.add_argument('-samples_folder', type=str)
        parser.add_argument('-model_folder', type=str)
        parser.add_argument('-output_size', type=int, default=1024)
        parser.add_argument('-gpu_num', nargs='?', const='0', default='0', type=str)

        args = parser.parse_args(sys.argv[2:])    

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
        from deepzine import DeepZine

        gan = DeepZine(inference=True,
                        gan_output_size=64,
                        inference_model_directory='~/DeepZine/data/log',
                        inference_output_directory='~/DeepZine/data/inference')

        gan.execute()


# def main():
    # Segment_GBM_cli()


def example_inference(gpu_num=0, base_directory='~/DeepZine/data/'):

    os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
    from deepzine import DeepZine

    gan = DeepZine(inference=True,
                    gan_output_size=64,
                    inference_model_directory='~/DeepZine/data/log',
                    inference_output_directory='~/DeepZine/data/inference')

    gan.execute()

    return


def example_interpolation(gpu_num=0, base_directory='~/DeepZine/data/'):

    os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
    from deepzine import DeepZine

    gan = DeepZine(interpolation=True,
                    gan_output_size=64,
                    inference_model_directory='~/DeepZine/data/log',
                    inference_output_directory='~/DeepZine/data/interpolation')

    gan.execute()

    return


def example_pipeline():

    return

if __name__ == '__main__':

    pass