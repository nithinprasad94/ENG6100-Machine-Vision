#import numpy as np
import os
from os import listdir
#from os.path import isfile, join
import cv2

class Single_Image():

    def __init__(self,img_path,img):
        self.img_path = img_path
        self.orig_img = img
        self.img_cropped = None
        self.img_PCA = None

    def add_cropped_img(self,img_cropped):
        self.img_cropped = img_cropped

    def add_PCA_img(self,img_PCA):
        self.img_PCA = img_PCA

class Image_Collection():

    def __init__(self):
        self.img_obj_list = []
        self.img_count = 0 #DEPRECATED

    def add_image_to_list(self,img_path,img):
        single_img_obj = Single_Image(img_path,img)
        self.img_obj_list.append(single_img_obj)
        self.img_count += 1 #DEPRECATED

    def return_orig_img_list(self):
        ret_list = []

        for i in range(len(self.img_obj_list)):
            orig_img = (self.img_obj_list[i]).orig_img
            ret_list.append(orig_img)

        return ret_list

    def return_cropped_img_list(self):
        ret_list = []

        for i in range(len(self.img_obj_list)):
            cropped_img = self.img_obj_list[i].cropped_img
            ret_list.append(cropped_img)

        return ret_list

    def crop_all_imgs(self,xmin,xmax,ymin,ymax):

        for i in range(len(self.img_obj_list)):
            cropped_img = self.img_obj_list[i].orig_img[xmin:xmax,ymin:ymax]
            self.img_obj_list[i].add_cropped_img(cropped_img)

class File_Reader():
    '''
    The FileReader class takes in a directory name and groups
    all the image files into respective data class objects.
    '''

    def __init__(self,dirname,dataset_labels = []):
        '''
        :param dirname: directory name to extract images from
        :param dataset_labels: should contain a category for each image type

        This function initializes the class with the directory name and creates empty lists of associated subdirectories
        and image filenames.
        '''
        self.dirname = dirname
        self.subdirs = []
        self.image_fnames = []
        self.images = []

    def enumerate_subdirs(self):
        '''
        This function gets subdirectories the parent directory and stores it into the self.subdirs variable (extends it)
        '''
        if self.subdirs != []:
            self.subdirs = []

        subdirs = listdir("./" + self.dirname)
        self.subdirs.extend(subdirs[:])

    def enumerate_fnames_from_subdirs(self):
        '''
        This function iterates through all the subdirectories, gets the image filenames for each subdirectory and appends
        it to the self.image_fnames list.
        '''
        for subdir in self.subdirs:
            currpath = "./" + self.dirname + "/" + subdir + "/"
            self.fnames = listdir("./")
            image_filenames = listdir(currpath)

            for fname in image_filenames:
                full_filepath = currpath + fname
                self.image_fnames.append(full_filepath)

                orig_img = cv2.imread(full_filepath)
                self.images.append(orig_img)

        #print(self.image_fnames[0:4])
        #print(self.images[0:4])

    def get_fnames_from_dir(self):
        '''
        gets subdirectories in a directory, and then gets all filenames from
        '''

        self.enumerate_subdirs()
        self.enumerate_fnames_from_subdirs()

        #print(self.image_fnames)
        #print(len(self.image_fnames))

        img_dataset = Image_Collection()

        for i in range(len(self.image_fnames)):
            #print("Image Names [i]:",self.image_fnames[i])
            img_dataset.add_image_to_list(self.image_fnames[i],self.images[i])
        return img_dataset

    def display_img(self,fname,fdescription="Input Image"):
        img = cv2.imread(fname)
        cv2.imshow(fdescription,img)
        cv2.waitKey(0)
        print(img.shape)
        new_img = cv2.resize(img,None,fx=0.5,fy=0.5)
        cv2.imshow(fdescription,new_img)
        cv2.waitKey(0)

fileObj = File_Reader("training_images")
dataBlock = fileObj.get_fnames_from_dir()
dataDict = dataBlock.img_obj_list
#print(dataDict.keys())
dataElem = dataDict[0]
#print(dataElem.keys())
orig_img = dataElem['original_img']
cropped_img = orig_img[0:800,0:800]
cv2.imshow("smaller img",cropped_img)
cv2.waitKey(0)

#fileObj.display_image('./training_images/150924030660/1_0.png')
#'./training_images/150924030660/1_0.png', './training_images/150924030660/1_1.png', './training_images/150924030660/1_10.png',
