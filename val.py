import os
import time
import torch
import json
import torch.optim as optim
from copy import deepcopy
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from torchvision.transforms import ToTensor, Normalize, Compose, ColorJitter
from net.loss import *
from net.network import CSPNet, CSPNet_mod
from config import Config
from dataloader.loader_val import *
from util.functions import parse_det_offset
from eval_city.eval_script.eval_demo import validate
from apex import amp
from memory_profiler import profile
import  argparse
import pdb

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--val', default='',type=str, metavar='PATH', 
        help='path to latest checkpoint (default: none)')
    # parser.add_argument('--wma', type=bool, default=False)
    args = parser.parse_args()
    return args

def main():
    cfg = Config()
    args = parse()

    print('Net Initializing')
    net = CSPNet().cuda()
    
    optimizer = optim.Adam(net.parameters(), lr=cfg.init_lr)
    amp.register_float_function(torch, 'sigmoid')
    net, optimizer = amp.initialize(net, optimizer, opt_level='O1')
    
    checkpoint = torch.load(args.val)
    net.load_state_dict(checkpoint)

    # dataset
    print('Dataset...')

    if cfg.val:
        testtransform = Compose(
        [ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        testdataset = CityPersons(path=cfg.train_path, type='val', config=cfg,
                                  caffemodel=cfg.caffemodel, preloaded=False)
        testloader = DataLoader(testdataset, batch_size=1, num_workers=0)

    MRs = val(testloader, net, cfg, args)

def val(testloader, net, config, args, teacher_dict=None):
    net.eval()
    if config.teacher:
        print('Load teacher params')
        student_dict = net.module.state_dict()
        net.module.load_state_dict(teacher_dict)
    print('Perform validation...')
    res = []
    t3 = time.time()
    for i, data in enumerate(testloader):
        inputs = data[0].cuda()
        y_center  = data[1]
        y_scale = data[2]
        with torch.no_grad():
            pos, height, offset = net(inputs)
        boxes = parse_det_offset(y_center[:,2,:,:].cpu().numpy(), y_scale[:,0,:,:].cpu().numpy(), offset[:,:2,:,:].cpu().numpy(), config.size_test, score=0.1, down=4, nms_thresh=0.5)
        # boxes = parse_det_offset(pos.cpu().numpy(), height.cpu().numpy(), offset.cpu().numpy(), config.size_test, score=0.1, down=4, nms_thresh=0.5)
        if len(boxes) > 0:
            boxes[:, [2, 3]] -= boxes[:, [0, 1]]

            for box in boxes:
                temp = dict()
                temp['image_id'] = i+1
                temp['category_id'] = 1
                temp['bbox'] = box[:4].tolist()
                temp['score'] = float(box[4])
                res.append(temp)

        print('\r%d/%d' % (i + 1, len(testloader)),end='')
        sys.stdout.flush()

    if config.teacher:
        print('\nLoad back student params')
        net.module.load_state_dict(student_dict)
    temp_val = './' + config.branch+'_temp_val.json'
    with open(temp_val, 'w') as f:
        json.dump(res, f)

    MRs = validate('./eval_city/val_gt.json', temp_val)
    t4 = time.time()
    print('Summerize: [Reasonable: %.2f%%], [Bare: %.2f%%], [Partial: %.2f%%], [Heavy: %.2f%%]'
          % (MRs[0]*100, MRs[1]*100, MRs[2]*100, MRs[3]*100))
    print('Validation time used: %.3f' % (t4 - t3))
    return MRs

def criterion(output, label, center, height, offset):
    cls_loss = center(output[0], label[0])
    reg_loss = height(output[1], label[1])
    off_loss = offset(output[2], label[2])
    return cls_loss, reg_loss, off_loss

def adjust_learning_rate(optimizer, epoch, config, args):
    if epoch < 3:
        lr = config.init_lr * float((epoch+1)*args.epoch_length)/(3.*args.epoch_length)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    if 3 <= epoch < config.lr_step[0]:
        for param_group in optimizer.param_groups:
            param_group['lr'] = config.init_lr

    if epoch in config.lr_step:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1

if __name__ == '__main__':
    main()

