import torch
import torch.nn as nn

class MaskedL1Loss(nn.Module):

	def __init__(self):
		super().__init__()

	def forward(self, pred, target):
		assert pred.dim() == target.dim(), "inconsistent dimensions"
		valid_mask = (target>0).detach()
		diff = target - pred
		diff = diff[valid_mask]
		self.loss = diff.abs().mean()
		return self.loss

class MaskedL2Loss(nn.Module):

	def __init__(self):
		super().__init__()

	def forward(self, pred, target):
		assert pred.dim() == target.dim(), "inconsistent dimensions"
		valid_mask = (target>0).detach()
		diff = target - pred
		diff = diff[valid_mask]
		self.loss = (diff ** 2).mean()
		return self.loss
