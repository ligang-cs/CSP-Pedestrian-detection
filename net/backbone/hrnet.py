import torch
import torch.nn as nn

from lib.hrnet import get_hrnet

import pdb

HRNet32 = dict(
	TYPE='hrnet_w32',
	FINAL_CONV_KERNEL=1,
	STAGE1=dict(
		NUM_MODULES=1,
		NUM_RANCHES=1,
		BLOCK='BOTTLENECK',
		NUM_BLOCKS=(4,),
		NUM_CHANNELS=(64,),
		FUSE_METHOD= 'SUM'),
	STAGE2=dict(
		NUM_MODULES=1,
		NUM_BRANCHES=2,
		BLOCK='BASIC',
		NUM_BLOCKS=(4, 4),
		NUM_CHANNELS=(32, 64),
		FUSE_METHOD='SUM'),
	STAGE3=dict(
		NUM_MODULES=4,
		NUM_BRANCHES=3,
		BLOCK='BASIC',
		NUM_BLOCKS=(4, 4, 4),
		NUM_CHANNELS=(32, 64, 128),
		FUSE_METHOD='SUM'),
	STAGE4=dict(
		NUM_MODULES=3,
		NUM_BRANCHES=4,
		BLOCK='BASIC',
		NUM_BLOCKS=(4, 4, 4, 4),
		NUM_CHANNELS=(32, 64, 128, 256),
		FUSE_METHOD='SUM'))

HRNet18 = dict(
	TYPE='hrnet_w18',
	FINAL_CONV_KERNEL=1,
	STAGE1=dict(
		NUM_MODULES=1,
		NUM_RANCHES=1,
		BLOCK='BOTTLENECK',
		NUM_BLOCKS=(4,),
		NUM_CHANNELS=(64,),
		FUSE_METHOD= 'SUM'),
	STAGE2=dict(
		NUM_MODULES=1,
		NUM_BRANCHES=2,
		BLOCK='BASIC',
		NUM_BLOCKS=(4, 4),
		NUM_CHANNELS=(18, 36),
		FUSE_METHOD='SUM'),
	STAGE3=dict(
		NUM_MODULES=4,
		NUM_BRANCHES=3,
		BLOCK='BASIC',
		NUM_BLOCKS=(4, 4, 4),
		NUM_CHANNELS=(18, 36, 72),
		FUSE_METHOD='SUM'),
	STAGE4=dict(
		NUM_MODULES=3,
		NUM_BRANCHES=4,
		BLOCK='BASIC',
		NUM_BLOCKS=(4, 4, 4, 4),
		NUM_CHANNELS=(18, 36, 72, 144),
		FUSE_METHOD='SUM'))

HRNet18_small = dict(
	TYPE='hrnet_w18_small',
	FINAL_CONV_KERNEL=1,
	STAGE1=dict(
		NUM_MODULES=1,
		NUM_RANCHES=1,
		BLOCK='BOTTLENECK',
		NUM_BLOCKS=(2,),
		NUM_CHANNELS=(64,),
		FUSE_METHOD= 'SUM'),
	STAGE2=dict(
		NUM_MODULES=1,
		NUM_BRANCHES=2,
		BLOCK='BASIC',
		NUM_BLOCKS=(2, 2),
		NUM_CHANNELS=(18, 36),
		FUSE_METHOD='SUM'),
	STAGE3=dict(
		NUM_MODULES=3,
		NUM_BRANCHES=3,
		BLOCK='BASIC',
		NUM_BLOCKS=(2, 2, 2),
		NUM_CHANNELS=(18, 36, 72),
		FUSE_METHOD='SUM'),
	STAGE4=dict(
		NUM_MODULES=2,
		NUM_BRANCHES=4,
		BLOCK='BASIC',
		NUM_BLOCKS=(2, 2, 2, 2),
		NUM_CHANNELS=(18, 36, 72, 144),
		FUSE_METHOD='SUM'))

class HRNet(nn.Module):
	def __init__(self, backbone):
		super(HRNet, self).__init__()
		self.hrnet_config = eval(backbone)
		self.hrnet = get_hrnet(self.hrnet_config)

	def forward(self, x):
		outputs = self.hrnet(x)
		return outputs
