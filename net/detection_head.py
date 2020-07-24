import torch
import torch.nn as nn
from .hough_module import Hough
import math
from torch.nn.modules.batchnorm import _BatchNorm
import pdb

class HoughNet_Head(nn.Module):
	"""using houghNet for center heatmap"""
	def __init__(self):
		super(HoughNet_Head, self).__init__()
		self.pos_conv = nn.Conv2d(256, 13, kernel_size=1)
		self.reg_conv = nn.Conv2d(256, 1, kernel_size=1)
		self.off_conv = nn.Conv2d(256, 2, kernel_size=1) 

		self.relu = nn.ReLU(inplace=True)

		self.houghNet = Hough(angle=90, R2_list=[4, 64, 256, 1024], num_classes=1,
                 region_num=13, vote_field_size=33)
	
	def init_weights(self):
		nn.init.xavier_normal_(self.pos_conv.weight)
		nn.init.xavier_normal_(self.reg_conv.weight)
		nn.init.xavier_normal_(self.off_conv.weight) 

		nn.init.constant_(self.reg_conv.bias, 0)
		# nn.init.constant_(self.pos_conv.bias, -math.log(0.99/0.01))
		nn.init.constant_(self.pos_conv.bias, 0)
		nn.init.constant_(self.off_conv.bias, 0)   

	def forward(self, x):

		evident_map = self.relu(self.pos_conv(x))
		score_map = self.houghNet(evident_map)
		x_cls = torch.sigmoid(score_map)
		x_reg = self.reg_conv(x)
		x_off = self.off_conv(x)
		return x_cls, x_reg, x_off


class CSP_Head(nn.Module):
	"""three sibling heads for cls, reg, offset"""
	def __init__(self):
		super(CSP_Head, self).__init__()
		self.head_ops = {'cls':1, 'reg': 1, 'offset': 2}

		for head in sorted(self.head_ops):
			num_output = self.head_ops[head]
			head_conv = nn.Sequential(
				nn.Conv2d(64, 256,
					kernel_size=3, padding=1, bias=True),
				nn.BatchNorm2d(256, momentum=0.01),
				nn.ReLU(inplace=True),
				nn.Conv2d(256, num_output,
					kernel_size=1))
			self.__setattr__(head, head_conv)

		self.init_weights()

	def init_weights(self):
		for head in self.head_ops:
			final_layers = self.__getattr__(head)
			for i, m in enumerate(final_layers.modules()):
				if isinstance(m, nn.Conv2d):
					if m.weight.shape[0] == self.head_ops[head]:
						if 'cls' in head:
							nn.init.xavier_normal_(m.weight)
							nn.init.constant_(m.bias, -math.log(0.99/0.01))
						else:
							nn.init.xavier_normal_(m.weight)
							nn.init.constant_(m.bias, 0)
					else:
						nn.init.xavier_normal_(m.weight)
						nn.init.constant_(m.bias, 0)
				elif isinstance(m, _BatchNorm):
					m.eval()

	def forward(self, x):
		output = {}
		for head in self.head_ops:
			if head == 'cls':
				output[head] = torch.sigmoid(self.__getattr__(head)(x))
			else:
				output[head] = self.__getattr__(head)(x)
		return output['cls'], output['reg'], output['offset']
		
