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
		
class L1LossSum(nn.Module):

	def __init__(self,Ld_ratio):
		super().__init__()
		self.ratio = Ld_ratio

	def forward(self, pred, target):
		assert pred.dim() == target.dim(), "inconsistent dimensions"
		#L1 depth loss
		valid_mask = (target>0).detach()
		diff = target - pred
		diff = diff[valid_mask]
		L_depth = diff.abs().mean()

		#L1 depth gradient loss
		L_grad = 0
		for h in [1,2,4,8,16]:
			pred_grad = calculate_depth_grad(pred, h)
			target_grad = calculate_depth_grad(target, h)
			valid_mask = (target_grad>0).detach()
			diff = target_grad - pred_grad
			diff = diff[valid_mask]
			L_grad += diff.abs().mean()

		return self.ratio*L_depth + (1 - self.ratio)*L_grad

def calculate_depth_grad(output,h):
	dims = output.dims()
	i_shift = torch.concat([output[:,:,h:,:],
				torch.zeros(dims[0],dims[1],dims[2]-h,dims[3])],1)
	i_norm = i_shift.abs().mean()
	j_shift = torch.concat([output[:,:,:,h:],
				torch.zeros(dims[0],dims[1],dims[2],dims[3]-h)],1)
	j_norm = j_shift.abs().mean()
	out_norm = output.abs().mean()

	g1 = (i_shift - output)/(i_norm + out_norm)
	g2 = (j_shift - output)/(j_norm + out_norm)

	return (g1, g2)
