# Adversarial-Images-Against-Super-Resolution-Convolutional-Neural-Networks-SRCNNs-for-Free
This repository is created for our paper "Adversarial Images Against Super-Resolution Convolutional Neural Networks (SRCNNs) for Free" which was published in PETS 2022.


Requirement: PyTorch, CUDA, robustness

**Steps befor running the codes:**

1. Download ImageNet Linf-norm eps=4 (ResNet50) and ImageNet L2-norm (ResNet50) eps=3 from [robustness github](https://github.com/MadryLab/robustness) and put them in models folder
2. Download CAR x2 and CAR x4 pretrained-models [from CAR github](https://github.com/sunwj/CAR). Rename them to CAR_x2.pt and CAR_x4.pt and move them to ./models folder.
3. Download RCAN models from [RCAN's official github](https://github.com/yulunzhang/RCAN) and put them in models directory

5. For running robust CNN, note that the CNNs are trained on ImageNet dataset. We selected 1000 of images randomly. If you want to select other set of random images in the code, download ImageNet data set and  provide the address of ImageNet directory (e.g., --imagenet_dir ./data/Imagenet). 

You can download all required pre-trained networks (robust models and CARs model ) and sampled data from [here](https://drive.google.com/drive/folders/1u-oD2kJDnnzOPhQSkfJJ1iKsEIRjt8VO?usp=sharing). To test the pipeline of robust CNN classifiers and SRCNNs, run RobustCNN.py as follows:

> python runRobustCNNs.py [--scale 2 or 4] [--filter blockaverage,bluring,medianBlur,bilateralFilter,None] [ --imagenet_dir ./data/Imagenet]

We downloaded robust CNNs from [robustness github](https://github.com/MadryLab/robustness) ( Check Robustness package here:\
@misc{robustness,\
   title={Robustness (Python Library)},\
   author={Logan Engstrom and Andrew Ilyas and Shibani Santurkar and Dimitris Tsipras},\
   year={2019},\
   url={https://github.com/MadryLab/robustness}\
})

**To run CAR and RCAN models:**

> python run_CAR.py --img_dir foldersofimages [--resize True/False] [--scale 2,4]

Note, if your images are LR images then --resize should be false otherwise we create  LR images by downscaling
 
