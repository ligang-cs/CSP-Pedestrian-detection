class Config(object):
    def __init__(self):
        # Select backbone from ['ResNet50', 'DLA34', 'HRNet18_small', 'HRNet18', 'HRNet32'] 
        self.backbone  = 'ResNet50' 

        # training config
        self.onegpu = 4
        self.num_epochs = 120
        self.add_epoch = 0
        self.iter_per_epoch = 2000 
        self.init_lr = 2e-4
        self.lr_policy = 'step'     #  or cyclic for SWA
        self.lr_step = [80]
        self.warm_up = 3
        self.alpha = 0.999
        # Set for SWA 
        # self.base_lr = 1e-4
        # self.target_ratio = (1, 0.1)

        # dataset
        self.root_path = '/ligang/Dataset/citypersons'     # the path to your citypersons dataset  

        # setting for data augmentation
        self.use_horizontal_flips = True
        self.brightness = (0.5, 2, 0.5)
        self.size_train = (640, 1280)
        self.size_test = (1024, 2048)    

        # image channel-wise mean to subtract, the order is BGR
        self.norm_mean = [123.675, 116.28, 103.53]
        self.norm_std = [58.395, 57.12, 57.375]

        # whether or not to perform validation during training
        self.val = True
        self.val_frequency = 2
        self.val_begin = 70

        # whether ot not to use the strategy of weight moving average 
        self.teacher = True     

        # TODO: adaptively adjust network  
        # setting for network architechture (Not implemented)
        self.point = 'center'  # or 'top', 'bottom
        self.scale = 'h'  # or 'w', 'hw'
        self.num_scale = 1  # 1 for height (or width) prediction, 2 for height+width prediction
        self.offset = True  # append offset prediction or not
        self.down = 4  # downsampling rate of the feature map for detection
        self.radius = 2  # surrounding areas of positives for the scale map

    def print_conf(self):
        print('\n'.join(['%s:%s' % item for item in self.__dict__.items()]))

    def write_conf(self, log):
        log.write('\n'.join(['%s:%s' % item for item in self.__dict__.items()]))