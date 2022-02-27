
import torch
import imageio

import torch.nn
from PIL import Image
from torchvision import transforms
import numpy as np

import utility
import model 
from ImageLoader import CustomDataset

import cv2
import os, argparse
import matplotlib



parser = argparse.ArgumentParser(description='Content Adaptive Resampler for Image downscaling')
parser.add_argument('--model_dir', type=str, default='./models', help='path to the pre-trained models')
parser.add_argument('--img_dir',default='./imgfolder', type=str, help='path to the HR images to be downscaled')
parser.add_argument('--scale', type=int,default=2,  help='downscale factor')
parser.add_argument('--output_dir', type=str, default='./results',  help='path to store results')
parser.add_argument('--resize', type=bool,default=True,  help='downscaling off/on if your images are LRs turn it off')

parser.add_argument('--save', type=str, default='RCAN',help='file name to save')
parser.add_argument('--model', default='RCAN',help='model name')
parser.add_argument('--load', type=str, default='.',help='file name to load')
parser.add_argument('--self_ensemble', action='store_true',help='use self-ensemble method for test')
parser.add_argument('--precision', type=str, default='single',choices=('single', 'half'),help='FP precision for test (single | half)')
parser.add_argument('--save_models', action='store_true',help='save all intermediate models')



args = parser.parse_args()

args.data_train='DIV2K'
args.data_test='DIV2K'

args.n_resgroups=10
args.n_resblocks=16
args.n_feats=64

args.kernel_size = 3
args.reduction = 16

scale = args.scale
args.n_colors=3
args.rgb_range=255
args.data_test=args.img_dir
args.test_only=True
args.chop=True
args.res_scale=1
args.print_model=False

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

tran_pil = transforms.ToPILImage()
cd = CustomDataset(args.data_test, scale=scale, resize=args.resize)
with torch.no_grad():
    args.scale=[scale]
    model = model.Model(args)
    model.eval()
    N=cd.__len__()
    for i in range(N):
        img,file_name= cd.__getitem__(i)
        img=img*255
        np_transpose = np.ascontiguousarray([img.transpose((2, 0, 1))])
        tensor = torch.from_numpy(np_transpose).float()
        d=model(tensor.cuda(), idx_scale=0)
        sr = utility.quantize(d, args.rgb_range)
        rec_img=sr[0].byte().permute(1, 2, 0).cpu().numpy()

        p=os.path.join(args.output_dir,file_name)
        matplotlib.image.imsave(p, rec_img)
        
