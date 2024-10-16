import argparse
import os
import torch

class BaseOptions():
    def __init__(self):
        self.initialized = False
        self.parser = None

    def initialize(self, parser):
        parser.add_argument('--mode', default='binary')
        parser.add_argument('--arch', type=str, default='res50', help='architecture for binary classification')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--resize_or_crop', type=str, default='scale_and_crop', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop|none]')
        parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal|xavier|kaiming|orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{loadSize}')
        parser.add_argument('--dataroot', type=str, required=True, help='path to the dataset')
        parser.add_argument('--classes', type=str, default='ai,nature', help='comma-separated list of class names')  # Add this line
        parser.add_argument('--rz_interp', type=str, nargs='+', default=['bilinear'], help='resize interpolation methods')
        parser.add_argument('--blur_sig', type=float, nargs=2, default=[0.1, 2.0], help='blur sigma range')
        parser.add_argument('--jpg_method', type=str, default='pil', help='jpeg compression method')
        parser.add_argument('--jpg_qual', type=int, nargs='+', default=[50, 95], help='jpeg quality range')
        parser.add_argument('--cropSize', type=int, default=224, help='crop size')
        parser.add_argument('--class_bal', action='store_true', help='if true, use class balancing')
        parser.add_argument('--loadSize', type=int, default=256, help='scale images to this size')
        parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
        parser.add_argument('--num_threads', type=int, default=8, help='number of data loading threads')
        parser.add_argument('--blur_prob', type=float, default=0.5, help='probability of applying blur')
        parser.add_argument('--jpg_prob', type=float, default=0.5, help='probability of applying jpeg compression')
        self.initialized = True
        return parser

    def gather_options(self):
        if not self.initialized:
            self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            self.parser = self.initialize(self.parser)

        opt, _ = self.parser.parse_known_args()
        return self.parser.parse_args()

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self, print_options=True):
        opt = self.gather_options()
        opt.isTrain = self.isTrain

        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        if print_options:
            self.print_options(opt)

        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        # Ensure classes is a list
        opt.classes = opt.classes if isinstance(opt.classes, list) else opt.classes.split(',')
        opt.rz_interp = opt.rz_interp if isinstance(opt.rz_interp, list) else opt.rz_interp.split(',')
        opt.blur_sig = opt.blur_sig if isinstance(opt.blur_sig, list) else opt.blur_sig.split(',')
        opt.blur_sig = [float(s) for s in opt.blur_sig]
        opt.jpg_method = opt.jpg_method if isinstance(opt.jpg_method, list) else opt.jpg_method.split(',')
        opt.jpg_qual = opt.jpg_qual if isinstance(opt.jpg_qual, list) else opt.jpg_qual.split(',')
        opt.jpg_qual = [int(s) for s in opt.jpg_qual]
        if len(opt.jpg_qual) == 2:
            opt.jpg_qual = list(range(opt.jpg_qual[0], opt.jpg_qual[1] + 1))
        elif len(opt.jpg_qual) > 2:
            raise ValueError("Shouldn't have more than 2 values for --jpg_qual.")

        self.opt = opt
        return self.opt


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def unnormalize(tens, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    return tens * torch.Tensor(std)[None, :, None, None] + torch.Tensor(mean)[None, :, None, None]
