import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from apex import amp

from net.detector import CSP
from config import Config
from dataloader.loader import *
from utils.functions import parse_det_offset
from eval_city.eval_script.eval_demo import validate

import json
import argparse
import pdb

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--val-path', default='',type=str, metavar='PATH', 
        help='path to latest checkpoint (default: none)')
    parser.add_argument('--json-out', default='',type=str,metavar='PATH',
        help='path to save detection results in json format')
    # parser.add_argument('--wma', type=bool, default=False)
    args = parser.parse_args()
    return args

def main():
    cfg = Config()
    args = parse()

    print('Net Initializing')
    net = CSP(cfg).cuda()
    
    optimizer = optim.Adam(net.parameters(), lr=cfg.init_lr)
    amp.register_float_function(torch, 'sigmoid')
    net, optimizer = amp.initialize(net, optimizer, opt_level='O1')
    net = nn.DataParallel(net)

    checkpoint = torch.load(args.val_path)
    net.module.load_state_dict(checkpoint['model'])

    # dataset
    print('Dataset...')

    if cfg.val:
        testdataset = CityPersons(path=cfg.root_path, type='val', config=cfg)
        testloader = DataLoader(testdataset, batch_size=1, num_workers=4)

    MRs = val(testloader, net, cfg, args)

def val(testloader, net, config, args, teacher_dict=None):
    net.eval()

    print('Perform validation...')
    res = []
    inference_time = 0
    num_images = len(testloader)
    for i, data in enumerate(testloader):
        inputs = data.cuda()
        with torch.no_grad():
            t1 = time.time()
            pos, height, offset = net(inputs)
            t2 = time.time()
            inference_time += (t2 - t1)

        boxes = parse_det_offset(pos.cpu().numpy(), height.cpu().numpy(), offset.cpu().numpy(), 
                            config.size_test, score=0.1, down=4, nms_thresh=0.5)
        if len(boxes) > 0:
            boxes[:, [2, 3]] -= boxes[:, [0, 1]]

            for box in boxes:
                temp = dict()
                temp['image_id'] = i+1
                temp['category_id'] = 1
                temp['bbox'] = box[:4].tolist()
                temp['score'] = float(box[4])
                res.append(temp)

        print('\r%d/%d' % (i + 1, num_images),end='')
        sys.stdout.flush()

    temp_val = args.json_out
    with open(temp_val, 'w') as f:
        json.dump(res, f)

    MRs = validate('./eval_city/val_gt.json', temp_val)
    print('\nSummerize:[Reasonable: %.2f%%], [Reasonable_small: %.2f%%], [Reasonable_occ=heavy: %.2f%%], [All: %.2f%%]'
          % (MRs[0]*100, MRs[1]*100, MRs[2]*100, MRs[3]*100))
    FPS = int(num_images / inference_time)
    print('FPS : {}'.format(FPS))
    return MRs

if __name__ == '__main__':
    main()

