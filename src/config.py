import time
import argparse

### config for NYU depth V2 dataset

parser = argparse.ArgumentParser(description='SDFORMER')

# Dataset
parser.add_argument('--dir_data',
                    type=str,
                    default='../nyudepthv2/',
                    help='path to dataset')
parser.add_argument('--data_name',
                    type=str,
                    default='NYU',
                    choices=('NYU', 'KITTIDC'),
                    help='dataset name')
parser.add_argument('--split_json',
                    type=str,
                    default='../data_json/nyu.json',
                    help='path to json file')
parser.add_argument('--patch_height',
                    type=int,
                    default=228,
                    help='height of a patch to crop')
parser.add_argument('--patch_width',
                    type=int,
                    default=304,
                    help='width of a patch to crop')
parser.add_argument('--top_crop',
                    type=int,
                    default=0,
                    help='top crop size for KITTI dataset')

# Hardware
parser.add_argument('--seed',
                    type=int,
                    default=7240,
                    help='random seed point')
parser.add_argument('--gpus',
                    type=str,
                    default="0,1",
                    help='visible GPUs')
parser.add_argument('--port',
                    type=str,
                    default='29500',
                    help='multiprocessing port')
parser.add_argument('--num_threads',
                    type=int,
                    default=1,
                    help='number of threads')
parser.add_argument('--no_multiprocessing',
                    action='store_true',
                    default=False,
                    help='do not use multiprocessing')

# Network
parser.add_argument('--model_name',
                    type=str,
                    default='SDFORMER',
                    choices='SDFORMER',
                    help='model name')
parser.add_argument('--window_sizes1',
                    default=[[12, 16], [6, 8], [4, 4]],
                    help='window_sizes')
parser.add_argument('--window_sizes2',
                    default=[[6, 19], [19, 8], [6, 4]],
                    help='window_sizes')
parser.add_argument('--window_sizes3',
                    default=[[3, 19], [19, 4], [3, 4]],
                    help='window_sizes')
parser.add_argument('--window_sizes4',
                    default=[[29, 38], [29, 19], [29, 2]],
                    help='window_sizes')
parser.add_argument('--inp_channels',
                    default=4,
                    help='inp_channels')
parser.add_argument('--out_channels',
                    default=1,
                    help='out_channels')
parser.add_argument('--dim',
                    default=24,
                    help='dim')
parser.add_argument('--num_blocks',
                    default=[2, 4, 6, 8],
                    help='num_blocks')
parser.add_argument('--heads',
                    default=[1, 2, 4, 8],
                    help='heads')
parser.add_argument('--num_refinement_blocks',
                    default=2,
                    help='num_refinement_blocks')
parser.add_argument('--ffn_expansion_factor',
                    default=2.88,
                    help='ffn_expansion_factor')
parser.add_argument('--bias',
                    default=False,
                    help='bias')
parser.add_argument('--LayerNorm_type',
                    default='WithBias',
                    help='LayerNorm_type')

# Training
parser.add_argument('--loss',
                    type=str,
                    default='1.0*L1+1.0*L2',
                    help='loss function configuration')
parser.add_argument('--opt_level',
                    type=str,
                    default='O0',
                    choices=('O0', 'O1', 'O2', 'O3'))
parser.add_argument('--pretrain',
                    type=str,
                    default=None,
                    help='ckpt path')
parser.add_argument('--resume',
                    action='store_true',
                    help='resume training')
parser.add_argument('--test_only',
                    action='store_true',
                    help='test only flag')
parser.add_argument('--epochs',
                    type=int,
                    default=25,
                    help='number of epochs to train')
parser.add_argument('--batch_size',
                    type=int,
                    default=8,
                    help='input batch size for training')
parser.add_argument('--max_depth',
                    type=float,
                    default=10.0,
                    help='maximum depth')
parser.add_argument('--augment',
                    type=bool,
                    default=True,
                    help='data augmentation')
parser.add_argument('--no_augment',
                    action='store_false',
                    dest='augment',
                    help='no augmentation')
parser.add_argument('--num_sample',
                    type=int,
                    default=500,
                    help='number of sparse samples')
parser.add_argument('--test_crop',
                    action='store_true',
                    default=False,
                    help='crop for test')

# Summary
parser.add_argument('--num_summary',
                    type=int,
                    default=4,
                    help='maximum number of summary images to save')

# Optimizer
parser.add_argument('--lr',
                    type=float,
                    default=0.0003,
                    help='learning rate')
parser.add_argument('--decay',
                    type=str,
                    default='10,15,20,25',
                    help='learning rate decay schedule')
parser.add_argument('--gamma',
                    type=str,
                    default='1.0,0.2,0.04,0.008',
                    help='learning rate multiplicative factors')
parser.add_argument('--optimizer',
                    default='ADAM',
                    choices=('SGD', 'ADAM', 'RMSprop'),
                    help='optimizer to use (SGD | ADAM | RMSprop)')
parser.add_argument('--momentum',
                    type=float,
                    default=0.9,
                    help='SGD momentum')
parser.add_argument('--betas',
                    type=tuple,
                    default=(0.9, 0.999),
                    help='ADAM beta')
parser.add_argument('--epsilon',
                    type=float,
                    default=1e-8,
                    help='ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay',
                    type=float,
                    default=0.0,
                    help='weight decay')
parser.add_argument('--warm_up',
                    action='store_true',
                    default=True,
                    help='do lr warm up during the 1st epoch')
parser.add_argument('--no_warm_up',
                    action='store_false',
                    dest='warm_up',
                    help='no lr warm up')

# Logs
parser.add_argument('--save',
                    type=str,
                    default='trial',
                    help='file name to save')
parser.add_argument('--save_full',
                    action='store_true',
                    default=False,
                    help='save optimizer, scheduler and amp in '
                         'checkpoints (large memory)')
parser.add_argument('--save_image',
                    action='store_true',
                    default=False,
                    help='save images for test')
parser.add_argument('--save_result_only',
                    action='store_true',
                    default=False,
                    help='save result images only with submission format')

args = parser.parse_args()

args.num_gpus = len(args.gpus.split(','))

current_time = time.strftime('%y%m%d_%H%M%S_')
save_dir = '../experiments/' + current_time + args.save
args.save_dir = save_dir

'''
import time
import argparse
# config for KITTI Depth Completion dataset

parser = argparse.ArgumentParser(description='SDFORMER')

# Dataset
parser.add_argument('--dir_data',
                    type=str,
                    default='../KITTI_DC',
                    help='path to dataset')
parser.add_argument('--data_name',
                    type=str,
                    default='KITTIDC',
                    choices=('NYU', 'KITTIDC'),
                    help='dataset name')
parser.add_argument('--split_json',
                    type=str,
                    default='../data_json/kitti_dc.json',
                    help='path to json file')
parser.add_argument('--patch_height',
                    type=int,
                    default=320,
                    help='height of a patch to crop')
parser.add_argument('--patch_width',
                    type=int,
                    default=1216,
                    help='width of a patch to crop')
parser.add_argument('--top_crop',
                    type=int,
                    default=20,
                    help='top crop size for KITTI dataset')

# Hardware
parser.add_argument('--seed',
                    type=int,
                    default=7240,
                    help='random seed point')
parser.add_argument('--gpus',
                    type=str,
                    default="0,1",
                    help='visible GPUs')
parser.add_argument('--port',
                    type=str,
                    default='29500',
                    help='multiprocessing port')
parser.add_argument('--num_threads',
                    type=int,
                    default=1,
                    help='number of threads')
parser.add_argument('--no_multiprocessing',
                    action='store_true',
                    default=False,
                    help='do not use multiprocessing')

# Network
parser.add_argument('--model_name',
                    type=str,
                    default='SDFORMER',
                    choices=('SDFORMER'),
                    help='model name')
parser.add_argument('--window_sizes1',
                    default=[[4, 4], [8, 8], [16, 16]],
                    help='window_sizes')
parser.add_argument('--window_sizes2',
                    default=[[4, 4], [8, 8], [16, 16]],
                    help='window_sizes')
parser.add_argument('--window_sizes3',
                    default=[[4, 4], [8, 8], [8, 16]],
                    help='window_sizes')
parser.add_argument('--window_sizes4',
                    default=[[4, 4], [4, 8], [4, 19]],
                    help='window_sizes')
parser.add_argument('--inp_channels',
                    default=4,
                    help='inp_channels')
parser.add_argument('--out_channels',
                    default=1,
                    help='out_channels')
parser.add_argument('--dim',
                    default=12,
                    help='dim')
parser.add_argument('--num_blocks',
                    default=[2, 2, 6, 8],
                    help='num_blocks')
parser.add_argument('--heads',
                    default=[1, 2, 4, 8],
                    help='heads')
parser.add_argument('--num_refinement_blocks',
                    default=2,
                    help='num_refinement_blocks')
parser.add_argument('--ffn_expansion_factor',
                    default=2.08,
                    help='ffn_expansion_factor')
parser.add_argument('--bias',
                    default=False,
                    help='bias')
parser.add_argument('--LayerNorm_type',
                    default='WithBias',
                    help='LayerNorm_type')

# Training
parser.add_argument('--loss',
                    type=str,
                    default='1.0*L1+1.0*L2',
                    help='loss function configuration')
parser.add_argument('--opt_level',
                    type=str,
                    default='O0',
                    choices=('O0', 'O1', 'O2', 'O3'))
parser.add_argument('--pretrain',
                    type=str,
                    default=None,
                    help='ckpt path')
parser.add_argument('--resume',
                    action='store_true',
                    help='resume training')
parser.add_argument('--test_only',
                    action='store_true',
                    help='test only flag')
parser.add_argument('--epochs',
                    type=int,
                    default=25,
                    help='number of epochs to train')
parser.add_argument('--batch_size',
                    type=int,
                    default=4,
                    help='input batch size for training')
parser.add_argument('--max_depth',
                    type=float,
                    default=90.0,
                    help='maximum depth')
parser.add_argument('--augment',
                    type=bool,
                    default=True,
                    help='data augmentation')
parser.add_argument('--no_augment',
                    action='store_false',
                    dest='augment',
                    help='no augmentation')
parser.add_argument('--num_sample',
                    type=int,
                    default=0,
                    help='number of sparse samples')
parser.add_argument('--test_crop',
                    action='store_true',
                    default=False,
                    help='crop for test')

# Summary
parser.add_argument('--num_summary',
                    type=int,
                    default=4,
                    help='maximum number of summary images to save')

# Optimizer
parser.add_argument('--lr',
                    type=float,
                    default=0.0002,
                    help='learning rate')
parser.add_argument('--decay',
                    type=str,
                    default='10,15,20,25',
                    help='learning rate decay schedule')
parser.add_argument('--gamma',
                    type=str,
                    default='1.0,0.2,0.04,0.008',
                    help='learning rate multiplicative factors')
parser.add_argument('--optimizer',
                    default='ADAM',
                    choices=('SGD', 'ADAM', 'RMSprop'),
                    help='optimizer to use (SGD | ADAM | RMSprop)')
parser.add_argument('--momentum',
                    type=float,
                    default=0.9,
                    help='SGD momentum')
parser.add_argument('--betas',
                    type=tuple,
                    default=(0.9, 0.999),
                    help='ADAM beta')
parser.add_argument('--epsilon',
                    type=float,
                    default=1e-8,
                    help='ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay',
                    type=float,
                    default=0.0,
                    help='weight decay')
parser.add_argument('--warm_up',
                    action='store_true',
                    default=True,
                    help='do lr warm up during the 1st epoch')
parser.add_argument('--no_warm_up',
                    action='store_false',
                    dest='warm_up',
                    help='no lr warm up')

# Logs
parser.add_argument('--save',
                    type=str,
                    default='trial',
                    help='file name to save')
parser.add_argument('--save_full',
                    action='store_true',
                    default=False,
                    help='save optimizer, scheduler and amp in '
                         'checkpoints (large memory)')
parser.add_argument('--save_image',
                    action='store_true',
                    default=False,
                    help='save images for test')
parser.add_argument('--save_result_only',
                    action='store_true',
                    default=False,
                    help='save result images only with submission format')

args = parser.parse_args()

args.num_gpus = len(args.gpus.split(','))

current_time = time.strftime('%y%m%d_%H%M%S_')
save_dir = '../experiments/' + current_time + args.save
args.save_dir = save_dir

'''
