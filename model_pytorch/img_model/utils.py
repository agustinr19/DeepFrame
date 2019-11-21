import os
import argparse

import torch

BASE_DIR = '/home/modfun/Desktop/sparse_to_dense_data'
data_choices = ['nyudepthv2', 'kitti', 'flightgoggles']
tov_choices = ['train', 'val']

parser = argparse.ArgumentParser()
# dataset arguments
parser.add_argument('-bdir', '--base_dir', type=str, default=BASE_DIR,
					help='The base directory of the dataset')
parser.add_argument('-ddir', '--data_dir', type=str, default='nyudepthv2', choices=data_choices,
					help='The dataset the model is trained/evaluated on')
parser.add_argument('-testdir', '--test_dir', type=str, default='',
					help='The dataset the model is tested on')
parser.add_argument('-odir', '--output_dir', type=str, default='/home/modfun/Desktop/deep_frame_output',
					help='The directory to the output models')
parser.add_argument('-task', '--task', type=str, default='train', choices=tov_choices,
					help='Specify whether the task is either train or validate')

# network/sampling parameters
parser.add_argument('-ns', '--n_samples', type=int, default=200,
					help='The number of samples for the uniform sampling function')
parser.add_argument('-md', '--max_depth', type=float, default=-1.0,
					help='The maximum depth for the uniform sampling function')
parser.add_argument('-enc', '--encoder', type=str, default='resnet50',
					help='The encoder for the network')
parser.add_argument('-dec', '--decoder', type=str, default='deconv3',
					help='The decoder for the network')
parser.add_argument('-dims', '--dims', type=str, default='rgb')
parser.add_argument('-os', '--output_size', type=int, default=(228, 304), nargs='+',
					help='The output size from the encoder/decoder network')
parser.add_argument('-pt', '--pre_trained', type=bool, default=True,
					help='Specify whether the model should be pretrained')
parser.add_argument('-ft_path', '--finetune_path', type=str, default='',
					help='Specify model path for finetuning the model')

loss_choices = ['l1', 'l2']
# optimization metrics
parser.add_argument('-bs', '--batch_size', type=int, default=10,
					help='The batch size per training batch')
parser.add_argument('-nw', '--n_workers', type=int, default=20,
					help='The number of workers for the data loader')
parser.add_argument('-ne', '--n_epochs', type=int, default=8,
					help='The number of epochs over the dataset')
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3,
					help='The learning rate for the network')
parser.add_argument('-mo', '--momentum', type=float, default=0.9,
					help='The momentum for the network')
parser.add_argument('-wd', '--weight_decay', type=float, default=1e-4,
					help='The weight decay for the network')
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
		args.encoder, args.decoder,
		args.dims,
		args.data_dir,
		str(args.n_samples),
		str(args.n_epochs), str(args.batch_size),
		str(args.learning_rate), str(args.momentum)
	])
	output_dir = os.path.join(args.output_dir, model_dir)
	if sub_dir:
		output_dir = os.path.join(output_dir, sub_dir)

	return output_dir

def modify_learning_rate(optimizer, epoch, lr):
	lr = lr * (0.1 ** (epoch // 5))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr

def save_checkpoint(state, epoch, output_dir):
	filename = os.path.join(output_dir, 'checkpoints', 'epoch-{}.tar'.format(epoch))
	torch.save(state, filename)
