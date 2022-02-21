# Adversarial-Images-Against-Super-Resolution-Convolutional-Neural-Networks-SRCNNs-for-Free
PETS 2022: Adversarial Images Against Super-Resolution Convolutional Neural Networks (SRCNNs) for Free


Requirement: PyTorch, CUDA, robustness

To test the pipeline of robust CNN classifiers and SRCNNs, run RobustCNN.py as follows:

python RobustCNN.py --scale [2 or 4] --filter [blockaverage,bluring,medianBlur,bilateralFilter,None]

Note that the CNNs are trained on ImageNet dataset. We selected 1000 of images randomly. If you want to select other set of random images in the code, download ImageNet data set and then uncomment Selecting_Candidate_Images() function first. Also you need to provide the address of ImageNet directory (e.g., --imagenet_dir ./data/Imagenet).



We downloaded robust CNNs from [robustness github](https://github.com/MadryLab/robustness)

@misc{robustness,
   title={Robustness (Python Library)},
   author={Logan Engstrom and Andrew Ilyas and Shibani Santurkar and Dimitris Tsipras},
   year={2019},
   url={https://github.com/MadryLab/robustness}
}

