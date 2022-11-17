# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import pprint

project_dir = Path(__file__).resolve().parent
dataset_dir = Path('/content/drive/MyDrive/A_Final_COGN.h5')
#dataset_dir = Path('/content/drive/MyDrive/Adversarial_Video_Summary-master/data/TVSum/eccv16_dataset_tvsum_google_pool5.h5')
#dataset_dir = Path('/content/drive/MyDrive/Adversarial_Video_Summary-master/data/SumMe/eccv16_dataset_summe_google_pool5.h5')
#dataset_dir = Path('/content/drive/MyDrive/A_Whole_cogn.h5')
#dataset_dir = Path('/content/drive/MyDrive/COGNIMUSE_2fps2.h5')
#video_list = ['360airballoon', '360parade', '360rowing', '360scuba', '360wedding']
save_dir = Path('/content/drive/MyDrive/SUM-GAN-AED/exp1')
#score_dir = Path('/content/drive/MyDrive/MSc/Adversarial_Video_Summary-master/results_scores')


def str2bool(v):
    """string to boolean"""
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class Config(object):
    def __init__(self, **kwargs):
        """Configuration Class: set kwargs as class attributes with setattr"""
        for k, v in kwargs.items():
            setattr(self, k, v)

        self.set_dataset_dir(self.video_type)

    def set_dataset_dir(self, video_type='cogn'):
        #if self.preprocessed:
        #    self.video_root_dir = dataset_dir.joinpath('resnet101_feature', video_type, self.mode)
        #else:
        #    self.video_root_dir = dataset_dir.joinpath('video_subshot', video_type, 'test')

        self.log_dir = save_dir.joinpath(video_type, 'logs/split'+str(self.split_index))
        self.score_dir = save_dir.joinpath(video_type, 'results/split'+str(self.split_index))
        self.save_dir = save_dir.joinpath(video_type, 'models/split'+str(self.split_index))

        #self.save_dir = save_dir.joinpath(video_type)
        #self.log_dir = self.save_dir
        #self.ckpt_path = self.save_dir.joinpath(f'epoch-{self.epoch}.pkl')
        #self.score_dir = score_dir

    def __repr__(self):
        """Pretty-print configurations in alphabetical order"""
        config_str = 'Configurations\n'
        config_str += pprint.pformat(self.__dict__)
        return config_str


def get_config(parse=True, **optional_kwargs):
    """
    Get configurations as attributes of class
    1. Parse configurations with argparse.
    2. Create Config class initilized with parsed kwargs.
    3. Return Config class.
    """
    parser = argparse.ArgumentParser()

    # Mode
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--verbose', type=str2bool, default='true')
    #parser.add_argument('--preprocessed', type=str2bool, default='True')
    parser.add_argument('--video_type', type=str, default='cogn')

    # Model
    parser.add_argument('--input_size', type=int, default=1024) #edw exei allo
    parser.add_argument('--hidden_size', type=int, default=500)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--summary_rate', type=float, default=0.3)

    # Train
    parser.add_argument('--n_epochs', type=int, default=50)
    parser.add_argument('--clip', type=float, default=5.0)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--discriminator_lr', type=float, default=1e-4)
    parser.add_argument('--discriminator_slow_start', type=int, default=1)
    parser.add_argument('--split_index', type=int, default=0)

    # load epoch
    parser.add_argument('--epoch', type=int, default=2)

    if parse:
        kwargs = parser.parse_args()
    else:
        kwargs = parser.parse_known_args()[0]

    # Namespace => Dictionary
    kwargs = vars(kwargs)
    kwargs.update(optional_kwargs)

    return Config(**kwargs)


if __name__ == '__main__':
    config = get_config()
    import ipdb
    ipdb.set_trace()
