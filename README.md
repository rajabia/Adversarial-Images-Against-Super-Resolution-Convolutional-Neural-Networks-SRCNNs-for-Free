# Adversarial-Images-Against-Super-Resolution-Convolutional-Neural-Networks-SRCNNs-for-Free
PETS 2022: Adversarial Images Against Super-Resolution Convolutional Neural Networks (SRCNNs) for Free


Requirement: PyTorch, CUDA, robustness


 

1. Download ImageNet Linf-norm eps=4 (ResNet50) and ImageNet L2-norm (ResNet50) eps=3 from [robustness github](https://github.com/MadryLab/robustness) and put them in models folder
2. Download CAR x2 and CAR  x4 pretrained-models [from CAR github](https://github.com/sunwj/CAR). Rename them to CARx2.pt and CARx4.pt and move them to ./models folder
3. Note that the CNNs are trained on ImageNet dataset. We selected 1000 of images randomly. If you want to select other set of random images in the code, download ImageNet data set and  provide the address of ImageNet directory (e.g., --imagenet_dir ./data/Imagenet). 

To test the pipeline of robust CNN classifiers and SRCNNs, run RobustCNN.py as follows:

python RobustCNN.py --scale [2 or 4] --filter [blockaverage,bluring,medianBlur,bilateralFilter,None]

We downloaded robust CNNs from [robustness github](https://github.com/MadryLab/robustness) ( Check Robustness package here: @misc{robustness,
   title={Robustness (Python Library)},
   author={Logan Engstrom and Andrew Ilyas and Shibani Santurkar and Dimitris Tsipras},
   year={2019},
   url={https://github.com/MadryLab/robustness}
})

