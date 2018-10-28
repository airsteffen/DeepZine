import os
import numpy as np

if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = str(0)

    from deepzine import DeepZine

    gan = DeepZine(output_size=64,
                    load_data=True,
                    train=True,
                    inference=False,
                    gan_output_size=64,
                    pdf_num=100,
                    download_pdf=False,
                    convert_pdf=False,
                    overwrite=False)

    # gan = DeepZine(progressive_depth=8,
    #                 train=False,
    #                 inference=False,
    #                 train_data_directory='/raw/Posterior',
    #                 train_hdf5='./rop_multiclass_with_cats.hdf5',
    #                 train_overwrite=False,
    #                 train_preloaded=True,
    #                 train_preprocess_images=False,
    #                 gan_samples_dir='/Test_Outputs/samples_rop_seg_and_rgb_plus_only',
    #                 gan_log_dir='./log_rop_seg_and_rgb_plus_only',
    #                 test_data_directory='./rop_same_latents',
    #                 test_model_path='./log_rop_masks/',
    #                 test_model_samples=200,
    #                 test_input_latent=input_latent)

    gan.execute()