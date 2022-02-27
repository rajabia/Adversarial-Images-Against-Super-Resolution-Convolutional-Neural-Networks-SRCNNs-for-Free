
import glob
import cv2
import numpy as np
# from option import args

class CustomDataset():
    def __init__(self,path, resize=True, scale=2):
        self.imgs_path = path
        self.file_list = glob.glob(self.imgs_path + "/*.jpg")
        self.file_list = self.file_list+glob.glob(self.imgs_path + "/*.jpeg")
        self.file_list = self.file_list+glob.glob(self.imgs_path + "/*.JPEG")
        self.file_list = self.file_list+glob.glob(self.imgs_path + "/*.JPG")
        self.file_list = self.file_list+glob.glob(self.imgs_path + "/*.PNG")
        self.file_list = self.file_list+glob.glob(self.imgs_path + "/*.png")
        self.resize=resize
        self.scale=scale
        print("Found %d  image(s) in %s " %(len(self.file_list), path)) 
        
            
    def __len__(self):
        return len(self.file_list) 

    def __getitem__(self, idx):
        img_path = self.file_list[idx]

        img = cv2.imread(img_path)
        
        if self.resize:
            width = int(img.shape[1] / self.scale)
            height = int(img.shape[0]/ self.scale)
            dim = (width, height)
            img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        file_name = img_path.split('/')

        img = img.astype(np.float)/255.0


        return img, file_name[-1]
