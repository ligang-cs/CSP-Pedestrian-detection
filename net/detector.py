import torch
import torch.nn as nn
from .backbone import DLA_34, ResNet_concat, HRNet
from .detection_head import  CSP_Head
import pdb


class CSP(nn.Module):
	"""CSP framework"""
	def __init__(self, cfg):
		super(CSP, self).__init__()
		backbone = cfg.backbone
		assert backbone is not None, 'Backbone must be specified.'
		if backbone == 'DLA34':
			self.backbone = DLA_34()
		elif backbone == 'ResNet50':
			self.backbone = ResNet_concat()
		elif 'HRNet' in backbone:
			self.backbone = HRNet(backbone)
		self.head = CSP_Head()

		self.init_weights()

	def init_weights(self):
		# self.backbone.init_weights()
		self.head.init_weights()

	def forward(self, x):
		features = self.backbone(x)
		cls_map, reg_map, off_map = self.head(features)
		return cls_map, reg_map, off_map
		