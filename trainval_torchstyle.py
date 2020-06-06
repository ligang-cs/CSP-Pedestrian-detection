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
from dataloader.loader import *
from util.functions import parse_det_offset
from eval_city.eval_script.eval_demo import validate
from apex import amp
# from memory_profiler import profile
import  argparse
import pdb

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', default='',type=str, metavar='PATH', 
        help='path to latest checkpoint (default: none)')
    # parser.add_argument ('--local_rank', type=int, default=0)
    # parser.add_argument('--wma', type=bool, default=False)
    args = parser.parse_args()
    return args

def main():
    cfg = Config()
    args = parse()
    # local_rank  = args.local_rank  

    print('Net Initializing')
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    net = CSPNet().cuda(cfg.gpu_ids[0])

    center = cls_pos().cuda()
    height = reg_pos().cuda()
    offset = offset_pos().cuda()

    # optimizer
    # params = []
    # for n, p in net.named_parameters():
    #     if p.requires_grad:
    #         params.append({'params': p})
    #     else:
    #         print(n)

    if cfg.teacher:
        teacher_dict = net.state_dict()
    else:
        teacher_dict = None

    # net = net.cuda()

    optimizer = optim.Adam(net.parameters(), lr=cfg.init_lr)
    amp.register_float_function(torch, 'sigmoid')
    net, optimizer = amp.initialize(net, optimizer, opt_level='O1')

    net = nn.DataParallel(net, device_ids=cfg.gpu_ids)

    if args.resume:
        def resume():
            if os.path.isfile(args.resume):
                print("=>loading checkpoint'{}'".format(args.resume))
                checkpoint = torch.load(args.resume)
                args.start_epoch = checkpoint['epoch']
                net.module.load_state_dict(checkpoint['model'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                amp.load_state_dict(checkpoint['amp'])
                print("=>loaded checkpoint '{}'(epoch {})"
                    .format(args.resume, checkpoint['epoch']))
                teacher_path = args.resume + '.tea'
                teacher = torch.load(teacher_path)
                teacher_dict.update(teacher)
            else:
                print("=>no checkpoint found at '{}'".format(args.resume))
        resume()
    else:
        args.start_epoch = 0

    # dataset
    print('Dataset...')
    batchsize = cfg.onegpu * len(cfg.gpu_ids)
    args.epoch_length = int(cfg.iter_per_epoch / batchsize)
    # traintransform = Compose(
    #     [ColorJitter(brightness=0.5), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    traindataset = CityPersons(path=cfg.train_path, type='train', config=cfg,
                              caffemodel=cfg.caffemodel)
    trainloader = DataLoader(traindataset, batch_size=batchsize, shuffle=True, num_workers=8)

    if cfg.val:
        testtransform = Compose(
        [ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        testdataset = CityPersons(path=cfg.train_path, type='val', config=cfg,
                                  caffemodel=cfg.caffemodel, preloaded=False)
        testloader = DataLoader(testdataset, batch_size=1, num_workers=2)

    cfg.print_conf()
    print('Training start')
    
    args.iter_num = args.epoch_length*cfg.num_epochs

    if not os.path.exists(cfg.ckpt_path):
        os.mkdir(cfg.ckpt_path)
    if not os.path.exists('./log'):
        os.mkdir('./log')

    # open log file
    log_file = os.path.join(cfg.ckpt_path,  cfg.branch + '.log')
    log = open(log_file, 'w')

    args.best_loss = np.Inf
    args.best_loss_epoch = 0
    args.best_mr = 100
    args.best_mr_epoch = 0

    if args.resume:
        args.iter_cur = args.start_epoch * args.epoch_length
    else:
        args.iter_cur = 0

    for epoch in range(args.start_epoch, cfg.num_epochs):
        print('----------')
        print('Epoch %d begin' % (epoch + 1))
        epoch_loss = train(trainloader, net, criterion, center, height, offset, optimizer, epoch, cfg, args, teacher_dict=teacher_dict)
        if cfg.val and (epoch + 1) >= cfg.val_begin and (epoch + 1) % cfg.val_frequency == 0:
            cur_mr = val(testloader, net, cfg, args, teacher_dict=teacher_dict)
            if cur_mr[0] < args.best_mr:
                args.best_mr = cur_mr[0]
                args.best_mr_epoch = epoch + 1
                print('Epoch %d has lowest MR: %.7f' % (args.best_mr_epoch, args.best_mr))
                log.write('epoch_num: %d loss: %.7f Summerize: [Reasonable: %.2f%%], [Reasonable_small: %.2f%%], [Reasonable_occ=heavy: %.2f%%], [All: %.2f%%], lr: %.6f\n'
                    % (epoch+1, epoch_loss, cur_mr[0]*100, cur_mr[1]*100, cur_mr[2]*100, cur_mr[3]*100, args.lr))
            else:
                print('Epoch %d has lowest MR: %.7f' % (args.best_mr_epoch, args.best_mr))
                log.write('epoch_num: %d loss: %.7f  lr: %.6f\n' % (epoch+1, epoch_loss, args.lr))
        if epoch+1 >= cfg.val_begin:
            print('Save checkpoint...')
            filename = cfg.ckpt_path + '/%s-%d.pth' % (net.module.__class__.__name__, epoch+1)
            checkpoint = {
            'epoch': epoch+1,
            'optimizer': optimizer.state_dict(),
            'amp': amp.state_dict()
            }
            if cfg.teacher:
                checkpoint['model'] = teacher_dict
            else:
                checkpoint['model'] = net.module.state_dict()
            torch.save(checkpoint, filename)
            # if cfg.teacher:
            #     torch.save(teacher_dict, filename+'.tea')
            print('%s saved.' % filename)
    log.close()

def train(trainloader, net, criterion, center, height, offset, optimizer, epoch, config, args, teacher_dict=None):
    t1 = time.time() 
    epoch_loss = 0.0
    total_loss_log, loss_cls_log, loss_reg_log, loss_offset_log, time_batch = 0, 0, 0, 0 ,0
    net.train()
    adjust_learning_rate(optimizer, epoch, config, args)
    args.lr = optimizer.param_groups[0]['lr']
    for i, data in enumerate(trainloader):   
        args.iter_cur += 1
        t3 = time.time()
        inputs, labels = data
        inputs = inputs.cuda()
        labels = [l.cuda().float() for l in labels]

        # zero the parameter gradients
        optimizer.zero_grad()

        # heat map
        outputs = net(inputs)

        # loss
        cls_loss, reg_loss, off_loss = criterion(outputs, labels, center, height, offset)
        loss = cls_loss + reg_loss + off_loss

        # loss.backward()
        with amp.scale_loss(loss, optimizer) as scale_loss:
            scale_loss.backward()

        # update param
        optimizer.step()

        if config.teacher:
            for k, v in net.module.state_dict().items():
                if k.find('num_batches_tracked') == -1:
                    teacher_dict[k] = config.alpha * teacher_dict[k] + (1 - config.alpha) * v
                else:
                    teacher_dict[k] = 1 * v

        # print statistics
        batch_loss = loss.item()
        batch_cls_loss = cls_loss.item()
        batch_reg_loss = reg_loss.item()
        batch_off_loss = off_loss.item()
        total_loss_log += batch_loss
        loss_cls_log += batch_cls_loss
        loss_reg_log += batch_reg_loss
        loss_offset_log += batch_off_loss

        t4 = time.time()
        time_batch += (t4-t3)
        if args.iter_cur % 20 == 0:
            ETA_time = (args.iter_num-args.iter_cur) * (time_batch/20)
            m ,s = divmod(ETA_time, 60)
            h, m =divmod(m, 60)
            print('\r[Epoch %d/%d, Batch %d/%d]$ <Total loss: %.6f> cls: %.6f, reg: %.6f, off: %.6f, Time: %.3f, lr:%.6f, ETA: %d:%02d:%02d' %
                (epoch + 1, config.num_epochs, i + 1, args.epoch_length,total_loss_log/20, loss_cls_log/20, loss_reg_log/20, loss_offset_log/20, time_batch/20, args.lr, h, m , s),end='')
            total_loss_log, loss_cls_log, loss_reg_log, loss_offset_log, time_batch = 0, 0, 0, 0, 0
        epoch_loss += batch_loss
        if i+1 == args.epoch_length:
            t2 = time.time()
            epoch_loss /= len(trainloader)
            print('\rEpoch %d end, AvgLoss is %.6f, Time used %.1f sec.' % (epoch+1, epoch_loss, int(t2-t1)))
            if epoch_loss < args.best_loss:
                args.best_loss = epoch_loss
                args.best_loss_epoch = epoch + 1
            print('Epoch %d has lowest loss: %.7f' % (args.best_loss_epoch, args.best_loss))
            break
    return epoch_loss

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

