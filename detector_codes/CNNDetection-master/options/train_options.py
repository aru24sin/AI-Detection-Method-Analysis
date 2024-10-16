from .base_options import BaseOptions
import argparse

class TrainOptions(BaseOptions):
    def __init__(self):
        super().__init__()
        self.isTrain = True  # Set isTrain to True for training options

    def initialize(self, parser):
        parser = super().initialize(parser)  # Call BaseOptions' initialize method
        parser.add_argument('--train_split', type=str, default='train', help='name of the train data directory')
        parser.add_argument('--val_split', type=str, default='val', help='name of the validation data directory')
       # parser.add_argument('--classes', nargs='+', default=['ai', 'nature'], help='list of class names')
       # parser.add_argument('--cropSize', type=int, default=224, help='crop size')
        #parser.add_argument('--loadSize', type=int, default=256, help='load size')
        #parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
        parser.add_argument('--niter', type=int, default=100, help='number of epochs to train for')
        parser.add_argument('--earlystop_epoch', type=int, default=10, help='early stopping patience')
        #parser.add_argument('--num_threads', type=int, default=8, help='number of data loading threads')
        #parser.add_argument('--blur_prob', type=float, default=0.5, help='probability of applying blur')
        #parser.add_argument('--blur_sig', type=float, nargs=2, default=[0.1, 2.0], help='blur sigma range')
        #parser.add_argument('--jpg_prob', type=float, default=0.5, help='probability of applying jpeg compression')
        #parser.add_argument('--jpg_method', type=str, default='pil', help='jpeg compression method')
        #parser.add_argument('--jpg_qual', type=int, nargs='+', default=[50, 95], help='jpeg quality range')
        #parser.add_argument('--rz_interp', type=str, nargs='+', default=['bilinear'], help='resize interpolation methods')
        parser.add_argument('--loss_freq', type=int, default=100, help='frequency of logging loss')
        parser.add_argument('--save_latest_freq', type=int, default=1000, help='frequency of saving the latest model')
        parser.add_argument('--save_epoch_freq', type=int, default=1, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--no_resize', action='store_true', help='if true, do not resize images')
        parser.add_argument('--no_crop', action='store_true', help='if true, do not crop images')
        parser.add_argument('--no_flip', action='store_true', help='if true, do not flip images')
        parser.add_argument('--serial_batches', action='store_true', help='if true, load images in order to make batches')
        #parser.add_argument('--class_bal', action='store_true', help='if true, use class balancing')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--optim', type=str, default='adam', help='optim to use [sgd, adam]')
        parser.add_argument('--beta1', type=float, default=0.9, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
        parser.add_argument('--data_aug', action='store_true', help='if specified, perform additional data augmentation (photometric, blurring, jpegging)')
        parser.add_argument('--new_optim', action='store_true', help='new optimizer instead of loading the optim state')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--last_epoch', type=int, default=-1, help='starting epoch count for scheduler intialization')

        return parser

    #def parse(self):
    #    parser = self.initialize(argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter))
    #    opt = parser.parse_args()
    #    opt.isTrain = self.isTrain  # Set the isTrain attribute
    #    self.parser = parser  # Assign the parser to self.parser
    #    self.print_options(opt)
    #    return opt
