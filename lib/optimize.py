from math import cos, pi

def adjust_learning_rate(optimizer, epoch, config, args):
    if config.lr_policy == 'step': 
        if epoch < config.warm_up:
            lr = config.init_lr * float((epoch+1) / config.warm_up)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        if config.warm_up <= epoch < config.lr_step[0]:
            for param_group in optimizer.param_groups:
                param_group['lr'] = config.init_lr

        if epoch in config.lr_step:
            for param_group in optimizer.param_groups:
                param_group['lr'] = config.init_lr * 0.1

    elif config.lr_policy == 'cyclic':
        # defalt: cyclic times = 1
        base_lr = config.base_lr
        start_ratio, end_ratio = config.target_ratio[0], \
                                                                config.target_ratio[1]
        progress = args.iter_cur % args.epoch_length
        lr = annealing_cos(base_lr * start_ratio, 
                                      base_lr * end_ratio,
                                      progress / (args.epoch_length))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

def annealing_cos(start, end, factor):
    assert 0 <= factor <= 1
    cos_out = cos(pi * factor) + 1
    return end + 0.5 * cos_out * (start - end)