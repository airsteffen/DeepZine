import sys
import os
import yaml


def process_config(config_dict):

    with open(config_dict, 'r') as stream:
        try:
            data_dict = yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = data_dict['gpu_num']

    from deepzine import DeepZine

    gan = DeepZine(**data_dict)

    gan.execute()

    return


if __name__ == '__main__':

    process_config(sys.argv[1])

    pass