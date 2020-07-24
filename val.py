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
from net.network import CSPNet, CSPNet_mod, CSPNet_DLA  
from config import Config
from dataloader.loader import *
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
    parser.add_argument('--json_out', default='',type=str,metavar='PATH',
        help='path to save detection results in json format')
    # parser.add_argument('--wma', type=bool, default=False)
    args = parser.parse_args()
    return args

def main():
    cfg = Config()
    args = parse()

    print('Net Initializing')
    net = CSPNet_DLA().cuda()
    
    optimizer = optim.Adam(net.parameters(), lr=cfg.init_lr)
    amp.register_float_function(torch, 'sigmoid')
    net, optimizer = amp.initialize(net, optimizer, opt_level='O1')
    net = nn.DataParallel(net)

    checkpoint = torch.load(args.val)
    net.module.load_state_dict(checkpoint['model'])

    # dataset
    print('Dataset...')

    if cfg.val:
        testdataset = CityPersons(path=cfg.train_path, type='val', config=cfg,
                                  caffemodel=cfg.caffemodel, preloaded=False)
        testloader = DataLoader(testdataset, batch_size=1, num_workers=0)

    MRs = val(testloader, net, cfg, args)

def val(testloader, net, config, args, teacher_dict=None):
    net.eval()

    print('Perform validation...')
    res = []
    t3 = time.time()
    for i, data in enumerate(testloader):
        inputs = data.cuda()
        with torch.no_grad():
            pos, height, offset = net(inputs)

        boxes = parse_det_offset(pos.cpu().numpy(), height.cpu().numpy(), offset.cpu().numpy(), config.size_test, score=0.1, down=4, nms_thresh=0.5)
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

    temp_val = args.json_out
    with open(temp_val, 'w') as f:
        json.dump(res, f)

    MRs = validate('./eval_city/val_gt.json', temp_val)
    t4 = time.time()
    print('Summerize:[Reasonable: %.2f%%], [Reasonable_small: %.2f%%], [Reasonable_occ=heavy: %.2f%%], [All: %.2f%%]'
          % (MRs[0]*100, MRs[1]*100, MRs[2]*100, MRs[3]*100))
    print('Validation time used: %.3f' % (t4 - t3))
    return MRs

if __name__ == '__main__':
    main()

