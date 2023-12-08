from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
from os import listdir
#from os.path import isfile, join
import cv2
import numpy as np

label_dict = {"type_55_nd":"Non-Defective", "type_55_d": "Defective", "type_55_pd":"Defective"}

class Single_Image():

    def __init__(self,img_path,img):
        self.img_path = img_path
        self.img_orig = img
        self.img_cropped = None
        self.img_cropped_gray = None
        self.img_pca_approx = None
        self.img_pca_features = None
        self.img_label = None

    def add_cropped_img(self,img_cropped):
        self.img_cropped = img_cropped

    def add_grayscale_img(self,img_grayscale):
        self.img_cropped_gray = img_grayscale

    def add_eigencomputations(self,new_features,img_pca_approx):
        self.img_pca_features = new_features
        self.img_pca_approx = img_pca_approx

    def set_label(self,inp_label):
        self.img_label = inp_label

    def display_img(self, img_format):
        #image = cv2.imread('img.jpg')
        if img_format == 0:
            image = cv2.cvtColor(self.img_orig,cv2.COLOR_BGR2RGB)
            plt.imshow(image)
            plt.title('Original Image')
            plt.axis('off')
            plt.show()

        elif img_format == 1:
            image = cv2.cvtColor(self.img_cropped,cv2.COLOR_BGR2RGB)
            plt.imshow(image)
            plt.title('Cropped Image')
            plt.axis('off')
            plt.show()

        elif img_format == 2:
            plt.imshow(self.img_cropped_gray,cmap='gray')
            plt.title('Grayscale Image')
            plt.axis('off')
            plt.show()

        elif img_format == 3:
            plt.imshow(self.img_pca_approx,cmap='gray')
            plt.title('PCA Image')
            plt.axis('off')
            plt.show()

        else:
            #image = cv2.cvtColor(self.img_cropped,cv2.COLOR_BGR2RGB)
            pass

class Image_Collection():

    def __init__(self):
        self.img_obj_list = []
        self.PCA_eigenvals = None
        self.PCA_eigenvecs_dim_d = None
        self.X = None
        self.target = None

    def add_image_to_list(self,img_path,img):
        single_img_obj = Single_Image(img_path,img)
        if "type_55_nd" in img_path:
            single_img_obj.set_label("ND")
        elif ("type_55_d" in img_path) or ("type_55_pd" in img_path):
            #NOTE: here we err on the side of caution and classify possible defects as defects until otherwise
            # determined by querying the SME (Subject Matter Expert) on the data.
            single_img_obj.set_label("D")
        self.img_obj_list.append(single_img_obj)

    def return_img_orig_list(self):
        ret_list = []

        for i in range(len(self.img_obj_list)):
            img_orig = (self.img_obj_list[i]).img_orig
            ret_list.append(img_orig)

        return ret_list

    def return_img_cropped_list(self):
        ret_list = []

        for i in range(len(self.img_obj_list)):
            cropped_img = self.img_obj_list[i].cropped_img
            ret_list.append(cropped_img)

        return ret_list
    def return_cropped_gray_img_list(self):
        ret_list = []

        for i in range(len(self.img_obj_list)):
            gray_img = self.img_obj_list[i].img_cropped_gray
            print("Gray Image Shape:",gray_img.shape)
            ret_list.append(gray_img)

        ret_list = np.array(ret_list)
        print("Gray List Array:",ret_list.shape)

        return ret_list

    def crop_all_imgs(self,xmin,xmax,ymin,ymax):

        for i in range(len(self.img_obj_list)):
            cropped_img = self.img_obj_list[i].img_orig[xmin:xmax,ymin:ymax]
            self.img_obj_list[i].add_cropped_img(cropped_img)

    def convert_cropped_to_grayscale(self):

        for i in range(len(self.img_obj_list)):
            img_grayscale = cv2.cvtColor(self.img_obj_list[i].img_cropped, cv2.COLOR_BGR2GRAY)
            self.img_obj_list[i].add_grayscale_img(img_grayscale)

    def return_PCA_features(self):
        ret_list = []

        for i in range(len(self.img_obj_list)):
            PCA_features = self.img_obj_list[i].img_pca_features
            ret_list.append(PCA_features)

        ret_list = np.array(ret_list)

        return ret_list

    def return_data_labels(self):
        ret_list = []

        for i in range(len(self.img_obj_list)):
            img_label = self.img_obj_list[i].img_label
            if img_label == "ND":
                ret_list.append(0) #Encode Non-defective data as 0
            elif img_label == "D":
                ret_list.append(1) #Encode defective data as 1
            else:
                ret_list.append(None)

        ret_list = np.array(ret_list)
        return ret_list

    def apply_PCA_on_all_images(self, n_components=110):
        PCA_def = PCA(n_components)

        vectorized_images = [] #Forms a n=110 * d=1.6302mil sized matrix
        for i in range(len(self.img_obj_list)):
            vectorized_img = np.reshape(self.img_obj_list[i].img_cropped_gray,-1)
            vectorized_images.append(vectorized_img)
        vectorized_images = np.array(vectorized_images)

        print("Vectorized Image Shape:",vectorized_images[0].shape)

        print("Pre-fit-transform")
        new_features = PCA_def.fit_transform(vectorized_images) #Coordinates in the lambda plane (plane of eigenvecs)
        print("Post-fit-transform")

        exp_var_pca = PCA_def.explained_variance_ratio_
        exp_var_cumul_sum = np.cumsum(exp_var_pca)
        print("Explained Variance Ratio for PCA: ",exp_var_pca)
        print("Explained variance Cumulative Sum for PCA: ",exp_var_cumul_sum)

        ###### < REFACTOR THIS CODE
        eigenvecs_dim_d = (PCA_def.components_).transpose() #Takes the row eigenvectors and turns it into column
        eigenvals = PCA_def.explained_variance_ #Should be 8 values
        print("Eigenvecs dim d shape:",eigenvecs_dim_d.shape) #Should be 1630200x8 sized matrix (implementation detail)

        # Create the visualization plot
        # REFERENCE: https://vitalflux.com/pca-explained-variance-concept-python-example/
        #plt.bar(range(0,len(exp_var_pca)), exp_var_pca, alpha=0.5, align='center', label='Individual explained variance')
        #plt.step(range(0,len(exp_var_cumul_sum)), exp_var_cumul_sum, where='mid',label='Cumulative explained variance')
        #plt.ylabel('Explained variance ratio')
        #plt.xlabel('Principal component index')
        #plt.legend(loc='best')
        #plt.tight_layout()
        #plt.show()
        ###### >

        #Add all the Eigenparams, as well as the Reconstructed Image using the new Eigenvectors
        self.PCA_eigenvals = eigenvals
        self.PCA_eigenvecs_dim_d = eigenvecs_dim_d

        for i in range(len(self.img_obj_list)):
            new_features_i = new_features[i].transpose()
            img_pca_approx = np.matmul(eigenvecs_dim_d,new_features_i)
            grayscale_dims = self.img_obj_list[i].img_cropped_gray.shape
            img_pca_approx = np.reshape(img_pca_approx,(grayscale_dims[0],grayscale_dims[1]))
            self.img_obj_list[i].add_eigencomputations(new_features[i],img_pca_approx)

        print("PCA Transformation Complete")

    def acquire_data(self,data):
        self.X = data
    def acquire_target(self,target):
        self.target = target

    def get_data_and_target(self):
        return (self.X,self.target)

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

                img_orig = cv2.imread(full_filepath)
                self.images.append(img_orig)

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

# fileObj = File_Reader("training_images")
# dataBlock = fileObj.get_fnames_from_dir()
# imgObjList = dataBlock.img_obj_list
#print(dataDict.keys())
# dataElem = dataDict[0]
#print(dataElem.keys())
# img_orig = dataElem['original_img']
# cropped_img = img_orig[0:800,0:800]
# cv2.imshow("smaller img",cropped_img)
# cv2.waitKey(0)

#fileObj.display_image('./training_images/150924030660/1_0.png')
#'./training_images/150924030660/1_0.png', './training_images/150924030660/1_1.png', './training_images/150924030660/1_10.png',
