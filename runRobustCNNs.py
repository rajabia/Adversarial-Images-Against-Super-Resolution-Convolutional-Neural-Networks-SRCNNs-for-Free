
print('This is Run Robust CNN file')
import os, argparse

import torch
import torch.nn as nn
from EDSR.edsr import EDSR
from modules import DSN
from robustness.datasets import ImageNet
from robustness.model_utils import make_and_restore_model
import pickle
import numpy as np
import torchvision
from torchvision import transforms
import math



#clean imagenet scale =4 array([0.92795657, 0.18377384, 0.4427025 ])
#clean imagenet scale =2 array([1.09009729, 0.20507661, 0.40530078])

parser = argparse.ArgumentParser(description='Content Adaptive Resampler for Image downscaling')
parser.add_argument('--model_dir', type=str, default='./models', help='path to the pre-trained model')
parser.add_argument('--scale', type=int,default=2,  help='downscale factor')
parser.add_argument('--output_dir', type=str, default='./results',  help='path to store results')
parser.add_argument('--imagenet_dir', type=str, default='None',  help='path to ImageNet Dataset')
parser.add_argument('--filter', type=str, default='None',  help='[ blockaverage,bluring,medianBlur,bilateralFilter,None]')




# Randomly choosing 1000 images that Their super-resolved LR are classified correctly
# by L_2tRobusr Imagenet Classifier (epsilon=3)
def Selecting_Candidate_Images(N=1000):
    #Loading the CAR SRCNN
    upscale_net.eval()
    torch.set_grad_enabled(False)


    ds = ImageNet(args.imagenet_dir)
    model_local, _ = make_and_restore_model(arch='resnet50',resume_path=args.model_dir+'/imagenet_l2_3_0.pt',dataset=ds)
    model_local.eval()

    model_target, _ = make_and_restore_model(arch='resnet50',resume_path=args.model_dir+'/imagenet_l2_3_0.pt',dataset=ds)
    model_local.eval()

    count=0
    images_candidate=[]
    Lables_candidates=[]
    scale=args.scale

    transform = transforms.Compose([transforms.ToTensor()]) #,transforms.RandomResizedCrop(224)
    imagenet_data = torchvision.datasets.ImageNet(args.imagenet_dir,split='val', transform=transform)
    data_loader = torch.utils.data.DataLoader(imagenet_data,batch_size=1,shuffle=True )
    print('data is Loaded')
    for i,(input_x, target) in enumerate(data_loader):


        

        if count==N:
            break
        # Check if the original image is classified correctly by Robust Imagenet CNN
        pred=model_local(input_x.cuda(), make_adv=False)
        pred=torch.argmax(pred[0][0])
        pred_1=pred.cpu()

        pred=model_target(input_x.cuda(), make_adv=False)
        pred=torch.argmax(pred[0][0])
        pred_2=pred.cpu()

        if pred_1==target and pred_2==target:


            
            # Creating LR images from Original HR
            downscaled_img=transforms.Resize(int(224/scale))(input_x)

            # Supper Resolving down-scaled images
            reconstructed_img = upscale_net(downscaled_img.cuda())
            reconstructed_img = torch.clamp(reconstructed_img, 0, 1)

            pred=model_local(reconstructed_img, make_adv=False)
            pred=torch.argmax(pred[0][0])
            pred=pred.cpu()
            print(len(Lables_candidates))
            if pred==target:
                count=count+1
                images_candidate.append(input_x)
                Lables_candidates.append(target)
                

    data={'x':images_candidate, 'y':Lables_candidates}
    Saved_Path=os.path.join(args.output_dir, "CandidateImageForRobustCNN_scale"+str(scale)+".pt")
    filehandler = open(Saved_Path,"wb")
    pickle.dump(data,filehandler)
    print("Candidate HR Images Saved to : " + Saved_Path)


# Calculating PSNR difference between two images
def calculate_psnr(img1, img2, Max_Value=255.0):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(Max_Value / math.sqrt(mse))

def Create_and_Test_Adversarial_Examples(eps=6, step_size=0.5, itrs=500,Local_CNN='imagenet_l2_3_0.pt', Target_CNN='imagenet_linf_4.pt', save_image=True):
    ds=ImageNet(args.imagenet_dir)

    #Loacding Local robust CNN
    local_model_path=os.path.join(args.model_dir, Local_CNN)
    local_model, _ = make_and_restore_model(arch='resnet50',resume_path=local_model_path,dataset=ds)
    local_model.eval()

    #Loacding target robust CNN
    target_model_path=os.path.join(args.model_dir, Target_CNN)
    target_model, _ = make_and_restore_model(arch='resnet50',resume_path=local_model_path,dataset=ds)
    target_model.eval()

    #Loading candidate images
    Saved_Path=os.path.join(args.output_dir, "CandidateImageForRobustCNN_scale"+str(scale)+".pt")
    filehandler = open(Saved_Path,"rb")
    data=pickle.load(filehandler)

    #attacks parameters
    attack_kwargs = {'constraint': '2', 'eps': eps,'step_size': step_size,'iterations': itrs,'targeted': False}
    
    #total_local: total number of adversarial could fool local CNN,
    #pert_local: amount of perturbation (PSNR) needed for fooling local CNN
    total_local,pert_local=0,0
    total_target,pert_target=0,0

    torch.set_grad_enabled(False)
    upscale_net.eval()
    
    #To convert Tensors to PIL images
    tran_pil = transforms.ToPILImage()

    for i in range(100,len(data['x'])):
        print("Proccessing  %d th image from %d images "%(i+1,len(data['x']) ))
        x=data['x'][i]
        y=data['y'][i]
        torch.set_grad_enabled(True)

        # Learning Adversarial image on Local CNN
        adv_out, adv_im = local_model(x.cuda() ,y.cuda(), make_adv=True, **attack_kwargs)
        y_new=torch.argmax(adv_out)
        
        #if the adversarial examples can fool local CNNs
        #Then we tested on the target robust CNN and measure minimum perturbation for
        #Fooling the Local CNN and target CNN
        psnr_res=calculate_psnr(np.round(adv_im.cpu().numpy()*255),np.round(x.cpu().numpy()*255))
        if not (y_new.cpu()==y):
            pert_local=pert_local+ psnr_res
            total_local=total_local+1

        adv_im=torch.reshape(adv_im,(1,3,224,224))
        
        torch.set_grad_enabled(False)
        downscaled_img=transforms.Resize(int(224/args.scale))(adv_im)
        downscaled_img=torch.reshape(downscaled_img,(1,3,int(224/args.scale),int(224/args.scale)))
        reconstructed_img = upscale_net(downscaled_img.cuda())
        reconstructed_img = torch.clamp(reconstructed_img, 0, 1)
        rec_img=reconstructed_img.cpu().numpy().reshape((224,224,3))

        if args.filter=='blockaverage':
            kernel = np.ones((3,3),np.float32)/9
            f_img= torch.reshape(torch.from_numpy(cv2.filter2D(rec_img,-1,kernel)),(1,3,224,224)).cuda()
        elif args.filter=='bluring':
            f_img= torch.reshape(torch.from_numpy(cv2.blur(rec_img,(3,3))),(1,3,224,224)).cuda()
        elif args.filter=='medianBlur':
            f_img= torch.reshape(torch.from_numpy(cv2.medianBlur(rec_img,3)),(1,3,224,224)).cuda()
        elif args.filter=='bilateralFilter':
            f_img= torch.reshape(torch.from_numpy(cv2.bilateralFilter(rec_img,3,75,75)),(1,3,224,224)).cuda()
        else:
            f_img=torch.reshape(torch.from_numpy(rec_img),(1,3,224,224)).cuda()

        y_target=target_model(f_img, make_adv=False)
        y_target=torch.argmax(y_target[0],axis=1)

        if not (y_target.cpu()==y):
            total_target=total_target+1
            pert_target=pert_target+psnr_res


        if save_image and i%100==0:
            c= int(np.floor(i/100)+1) 
            
            img_path=os.path.join(args.output_dir,"Original_Adversarial_image_"+str(c) +".png")
            img=tran_pil(adv_im[0])
            img.save(img_path)

            if args.filter!= 'None':
                img_path=os.path.join(args.output_dir,args.filter+"_Super_Resolved_"+str(c) +".png")
                img=tran_pil(f_img[0])
                img.save(img_path)

            img_path=os.path.join(args.output_dir,"Super_Resolved_Adversarial_image_"+str(c) +".png")
            img=tran_pil(torch.squeeze(reconstructed_img))
            img.save(img_path)

            img_path=os.path.join(args.output_dir,"original_image_"+str(c) +".png")
            img=tran_pil(torch.squeeze(data['x'][i]))
            img.save(img_path)
            
            



      
    print("Filter Used:" +args.filter) 
    print("Fooling Rate of Local CNN is %.4f and the average perturbation needed is %.4f (PSNR)"%(total_local/float(len(data['y'])), pert_local/float(len(data['y']))) )  
    print("Fooling Rate of target CNN is %.4f and the average perturbation needed is %.4f (PSNR)"%(total_target/float(len(data['y'])), pert_target/float(len(data['y']))) )  
        


if __name__ == '__main__':
    
    args = parser.parse_args()
    if args.imagenet_dir!= 'None'
        Selecting_Candidate_Images()

    KSIZE = 3 * args.scale + 1
    OFFSET_UNIT = args.scale

    kernel_generation_net = DSN(k_size=KSIZE, scale=args.scale).cuda()

    upscale_net = EDSR(32, 256, scale=args.scale).cuda()

    upscale_net = nn.DataParallel(upscale_net, [0])

    upscale_net.load_state_dict(torch.load(os.path.join(args.model_dir, 'CAR_x{0}'.format(args.scale)+'.pt')))


    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    Create_and_Test_Adversarial_Examples()
    
    



