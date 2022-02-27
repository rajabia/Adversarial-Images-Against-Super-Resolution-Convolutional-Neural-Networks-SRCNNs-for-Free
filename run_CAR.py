import os, argparse

import torch
import torch.nn as nn
from model.edsr import EDSR
from modules import DSN

import numpy as np
import torchvision
from torchvision import transforms
import math
from ImageLoader import CustomDataset



parser = argparse.ArgumentParser(description='Content Adaptive Resampler for Image downscaling')
parser.add_argument('--model_dir', type=str, default='./models', help='path to the pre-trained model')
parser.add_argument('--scale', type=int,default=2,  help='downscale factor')
parser.add_argument('--output_dir', type=str, default='./results',  help='path to store results')
parser.add_argument('--img_dir',default='./imgfolder', type=str, help='path to the HR/LR images to be downscaled')
parser.add_argument('--resize', type=bool,default=True,  help='downscaling off/on if your images are LRs turn it off')

args = parser.parse_args()


KSIZE = 3 * args.scale + 1
OFFSET_UNIT = args.scale

kernel_generation_net = DSN(k_size=KSIZE, scale=args.scale).cuda()

upscale_net = EDSR(32, 256, scale=args.scale).cuda()

upscale_net = nn.DataParallel(upscale_net, [0])

upscale_net.load_state_dict(torch.load(os.path.join(args.model_dir, 'CAR_x{0}'.format(args.scale)+'.pt')))
#To convert Tensors to PIL images
tran_pil = transforms.ToPILImage()

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)


cd = CustomDataset(args.img_dir, scale=args.scale, resize=False)
trans=transforms.ToTensor()

N=cd.__len__()
for i in range(N):
	img,file_name= cd.__getitem__(i)
	input_x=trans(np.array(img))

	if args.resize:
		# Creating LR images from Original HR
		downscaled_img=transforms.Resize([int(img.shape[0]/args.scale),int(img.shape[1]/args.scale)])(input_x)
		downscaled_img=torch.reshape(downscaled_img,(1,3,int(img.shape[0]/args.scale),int(img.shape[1]/args.scale)))
	else:
		downscaled_img=input_x
		downscaled_img=torch.reshape(downscaled_img,(1,3,int(img.shape[0]),int(img.shape[1])))


	# Supper Resolving down-scaled images
	reconstructed_img = upscale_net(downscaled_img.type(torch.FloatTensor).cuda())
	reconstructed_img = torch.clamp(reconstructed_img, 0, 1)
	p=os.path.join(args.output_dir,file_name)
	im.save(p)


