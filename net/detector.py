import torch
import torch.nn as nn
from .backbone import DLA_34
from .detection_head import HoughNet_Head, CSP_Head

class CSP(nn.Module):
	"""CSP framework"""
	def __init__(self):
		super(CSP, self).__init__()
		self.backbone = DLA_34()
		self.head = CSP_Head()

		self.init_weights()

	def init_weights(self):
		# self.backbone.init_weights()
		self.head.init_weights()

	def forward(self, x):
		features = self.backbone(x)
		cls_map, reg_map, off_map = self.head(features)
		return cls_map, reg_map, off_map
		