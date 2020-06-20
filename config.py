class Config(object):
    def __init__(self):
        self.branch = 'DLA-34-bs12'
        self.gpu_ids = [0, 1]
        self.onegpu = 6
        self.num_epochs = 120
        self.add_epoch = 0
        self.iter_per_epoch = 2000  # 2000
        self.init_lr = 2e-4
        self.lr_step = [80]
        self.alpha = 0.999

        # dataset
        self.train_path = '/ligang/Dataset/citypersons'
        self.test_path =  '/ligang/Dataset/citypersons'
        self.ckpt_prefix = '/ligang/Works/PedDet/CSP/Experiments/'
        self.ckpt_path = self.ckpt_prefix + self.branch
        self.train_random = False

        # setting for network architechture
        self.network = 'resnet50'  # or 'mobilenet'
        self.point = 'center'  # or 'top', 'bottom
        self.scale = 'h'  # or 'w', 'hw'
        self.num_scale = 1  # 1 for height (or width) prediction, 2 for height+width prediction
        self.offset = True  # append offset prediction or not
        self.down = 4  # downsampling rate of the feature map for detection
        self.radius = 2  # surrounding areas of positives for the scale map

        # setting for data augmentation
        self.use_horizontal_flips = True
        self.brightness = (0.5, 2, 0.5)
        self.size_train = (640, 1280)
        self.size_test = (1024, 2048)

        # image channel-wise mean to subtract, the order is BGR
        self.norm_mean = [123.675, 116.28, 103.53]
        self.norm_std = [58.395, 57.12, 57.375]

        # whether or not use caffe style training which is used in paper
        self.caffemodel = True

        # whether or not to do validation during training
        self.val = True
        self.val_frequency = 2
        self.val_begin = 70

        self.teacher = True     
        self.restore = False

    def print_conf(self):
        print('\n'.join(['%s:%s' % item for item in self.__dict__.items()]))