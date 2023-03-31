import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from sklearn.metrics import r2_score, mean_absolute_error
from utils.general import *
from utils.datagenerator import DataIterator
from scannet.layers import SGDRC
from scannet.models import SCANNet
import time
import argparse
import yaml
import random
import numpy as np
from tensorflow.keras.callbacks import *
import tensorflow as tf


def set_seed(seed=2134):
    # tf.keras.utils.set_random_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

def main(args):

    set_seed(2134)

    config = yaml.safe_load(open(args.dataset))
    print('Create model use Ring Information: ', args.use_ring)

    config['model']['use_ring'] = args.use_ring

    config['hyper']['use_ref'] = args.use_ref
    config['hyper']['target'] = args.target
    config['hyper']['pretrained'] = args.pretrained

    scannet = SCANNet(config)
    scannet.init_model(args.pretrained)

    print('Load data for dataset: ', args.dataset)
    scannet.prepare_dataset()

    print('Start training model')
    start = time.time()
    scannet.train(3)

    print('Training time: ', time.time()-start)

    # Evaluate for testset
    scannet.evaluate()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('target', type=str,
                        help='Target energy for training')

    parser.add_argument('dataset', type=str, help='Path to dataset configs')

    parser.add_argument('--use_ring', type=bool, default=False,
                        help='Whether to use ring as extra emedding')

    parser.add_argument('--use_ref', type=bool, default=False,
                        help='Whether to use ref optimization energy')

    parser.add_argument('--pretrained', type=str, default='',
                        help='Path to pretrained model (optional)')

    args = parser.parse_args()
    main(args)
