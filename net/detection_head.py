import torch
import torch.nn as nn
import math
from torch.nn.modules.batchnorm import _BatchNorm
import pdb

class CSP_Head(nn.Module):
	"""three sibling heads for cls, reg, offset"""
	def __init__(self):
		super(CSP_Head, self).__init__()
		self.head_ops = {'cls':1, 'reg': 1, 'offset': 2}

		for head in sorted(self.head_ops):
			num_output = self.head_ops[head]
			head_conv = nn.Sequential(
				# nn.Conv2d(64, 256,
				# 	kernel_size=3, padding=1, bias=True),
				# nn.BatchNorm2d(256, momentum=0.01),
				# nn.ReLU(inplace=True),
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
		
