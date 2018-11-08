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
                        pipeline               Run the entire pipeline.
                        load_data              Use the internetarchive API to download data.
                        train_model            
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

        parser.add_argument('-output_folder', type=str)
        parser.add_argument('-T1', type=str)
        parser.add_argument('-T1POST', type=str)
        parser.add_argument('-FLAIR', type=str)
        parser.add_argument('-input_directory', type=str)
        parser.add_argument('-wholetumor_output', nargs='?', type=str, const='wholetumor.nii.gz', default='wholetumor.nii.gz')
        parser.add_argument('-enhancing_output', nargs='?', type=str, const='enhancing.nii.gz', default='enhancing.nii.gz')
        parser.add_argument('-gpu_num', nargs='?', const='0', default='0', type=str)
        parser.add_argument('-debiased', action='store_true')  
        parser.add_argument('-resampled', action='store_true')
        parser.add_argument('-registered', action='store_true')
        parser.add_argument('-skullstripped', action='store_true') 
        parser.add_argument('-preprocessed', action='store_true') 
        parser.add_argument('-save_preprocess', action='store_true')
        parser.add_argument('-save_all_steps', action='store_true')
        parser.add_argument('-output_probabilities', action='store_true')
        args = parser.parse_args(sys.argv[2:])

    def pipeline(self):

        args = self.parse_args()

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)

        from deepneuro.pipelines.Segment_GBM.predict import predict_GBM

        predict_GBM(args.output_folder, FLAIR=args.FLAIR, T1POST=args.T1POST, T1PRE=args.T1, ground_truth=None, input_directory=args.input_directory, bias_corrected=args.debiased, resampled=args.resampled, registered=args.registered, skullstripped=args.skullstripped, preprocessed=args.preprocessed, save_preprocess=args.save_preprocess, save_all_steps=args.save_all_steps, output_wholetumor_filename=args.wholetumor_output, output_enhancing_filename=args.enhancing_output)


# def main():
    # Segment_GBM_cli()


def example_data_load(gpu_num=0, base_directory='~/DeepZine/data/'):

    os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
    from deepzine import DeepZine

    gan = DeepZine(load_data=True,
                    gan_output_size=64,
                    pdf_num=10000,
                    download_pdf=True,
                    convert_pdf=True,
                    overwrite=True,
                    collection='MBLWHOI',
                    pdf_directory=os.path.join(base_directory, 'pdfs'),
                    image_directory=os.path.join(base_directory, 'images'),
                    data_hdf5=os.path.join(base_directory, 'pages.hdf5')
                    )

    gan.execute()

    return


def example_train(gpu_num=0, base_directory='~/DeepZine/data/'):

    os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
    from deepzine import DeepZine

    gan = DeepZine(train=True,
                    gan_output_size=64)

    gan.execute()

    return


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