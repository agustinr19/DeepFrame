import os
import argparse

import torch

# BASE_DIR = '../../test_data/rgbd-scenes'
# data_choices = ['desk','background','kitchen_small','meeting_small','table','table_small']
BASE_DIR = '../../test_data/'
data_choices = ['rgbd-scenes']
tov_choices = ['train', 'val']
cnn_choices = ['single', 'stack']

parser = argparse.ArgumentParser()
# dataset arguments
parser.add_argument('-bdir', '--base_dir', type=str, default=BASE_DIR,
					help='The base directory of the dataset')
parser.add_argument('-ddir', '--data_dir', type=str, default=data_choices[0], choices=data_choices,
					help='The dataset the model is trained/evaluated on')
parser.add_argument('-testdir', '--test_dir', type=str, default='',
					help='The dataset the model is tested on')
# parser.add_argument('-odir', '--output_dir', type=str, default='../../../cnn_single_output',
# 					help='The directory to the output models')
parser.add_argument('-type', '--task', type=str, default='train', choices=tov_choices,
					help='Specify whether the task is either train or validate')
parser.add_argument('-cnn', '--cnn_type', type=str, default='single', choices=cnn_choices,
					help='Specify whether the architecture is either CNN-Single or CNN-Stack')
parser.add_argument('-ss', '--stack_size', type=int, default=10,
					help='If the architecture is CNN-Stack, the size of the stack')

loss_choices = ['l1', 'l2','l1sum']
# optimization metrics
parser.add_argument('-bs', '--batch_size', type=int, default=10,
					help='The batch size per training batch')
parser.add_argument('-nw', '--n_workers', type=int, default=20,
					help='The number of workers for the data loader')
parser.add_argument('-ne', '--n_epochs', type=int, default=8,
					help='The number of epochs over the dataset')
parser.add_argument('-lr', '--learning_rate', type=float, default=2e-4,
					help='The learning rate for the network')
parser.add_argument('-b', '--betas', type=tuple, default=(0.9,0.999),
                                        help='The beta parameters for Adam optimizer')
parser.add_argument('-ct', '--criterion', type=str, default='l1', choices=loss_choices,
					help='The loss function for the network')

# data output parameters
parser.add_argument('-pfreq', '--print_freq', type=int, default=20,
					help='The frequency with which to write to the log')
parser.add_argument('-valfq', '--val_frequency', type=int, default=100,
					help='The frequency with which to save validation images')

def parse_args():
	return parser.parse_args()

def fetch_output_dir(args, sub_dir = None):
	model_dir = '-'.join([
		args.data_dir,
		str(args.n_epochs), str(args.batch_size),
		str(args.learning_rate)
	])
	if args.cnn_type == 'single'
		output_start = '../../../cnn_single_output'
	else:
		output_start = '../../../cnn_stack_output'
	output_dir = os.path.join(output_start, model_dir)
	if sub_dir:
		output_dir = os.path.join(output_dir, sub_dir)

	return output_dir

def save_checkpoint(state, epoch, output_dir):
	filename = os.path.join(output_dir, 'checkpoints', 'epoch-{}.tar'.format(epoch))
	torch.save(state, filename)
