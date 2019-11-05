import os

from PIL import Image
import numpy as np

import matplotlib.pyplot as plt

def _save_image(img_name, np_img):
	np_img = np_img * 255
	img = Image.fromarray(np_img.astype('uint8'))
	img.save(img_name)

def _save_heatmap(img_name, np_img):
	cmap = plt.cm.jet
	norm = plt.Normalize(vmin=np_img.min(), vmax=np_img.max())
	heatmap = cmap(norm(np_img))
	plt.imsave(img_name, heatmap)

	# plt.imshow(np_img, cmap='jet', interpolation='nearest')
	# plt.show()

def save_heatmap(img_dir, img_name):
	depth_path = os.path.join(img_dir, img_name)
	depth_img = Image.open(depth_path)
	depth_img.load()
	
	np_img = np.asarray(depth_img, dtype='uint8')
	_save_heatmap(img_name, np_img)

def save_visualization(output_dir, index, network_input, network_output, network_target):
	network_output = network_output.squeeze(0)
	network_target = network_target.squeeze(0)
	vis_dir = os.path.join(output_dir, 'visualizations')

	batch_size = len(network_input)
	start_index = index * batch_size

	for (sub_index, data) in enumerate(network_input):
		rgb = network_input[sub_index,:3,:,:]
		depth_output = network_output[sub_index,:,:]
		depth_target = network_target[sub_index,:,:]

		rgb = rgb.permute(1, 2, 0)

		_save_image(os.path.join(vis_dir, 'rgb_{:05d}.png'.format(start_index + sub_index)), rgb.cpu().numpy())
		_save_heatmap(os.path.join(vis_dir, 'pred_{:05d}.png'.format(start_index + sub_index)), depth_output.cpu().numpy())
		_save_heatmap(os.path.join(vis_dir, 'depth_{:05d}.png'.format(start_index + sub_index)), depth_target.cpu().numpy())
