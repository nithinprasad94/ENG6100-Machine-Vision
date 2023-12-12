# Project:
# Authors: Helena Kunic, Nithin Prasad

import cv2
import preprocessing
import ML_models
import numpy as np

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # Complete all preprocessing steps
    fileObj = preprocessing.File_Reader("training_images")
    imgCollection = fileObj.get_fnames_from_dir()
    imgCollection.crop_all_imgs(450, 1750, 200, 1454)
    imgCollection.convert_cropped_to_grayscale()
    imgObjList = imgCollection.img_obj_list
    first_image = imgObjList[0]
    #print("Original Image Shape:",first_image.img_orig.shape)
    #print("Cropped Image Shape:",first_image.img_cropped.shape)
    #print("Grayscale Image Shape:",first_image.img_cropped_gray.shape)
    #first_image.display_img(0)
    #first_image.display_img(1)
    #first_image.display_img(2)
    print("Number of Images:",len(imgObjList))
    #first_image.display_img(0)
    #first_image.display_img(1)
    #first_image.display_img(0)
    #first_image.display_img(1)
    #first_image.display_img(2)

    # Apply PCA
    pca_vals = [19]
    for i in pca_vals:
        imgCollection.apply_PCA_on_all_images(i) #NUMBER OF PCA COMPONENTS: start at 19 components for 95% loss ... might still be missing defects
        first_image = imgObjList[0]
        print("First Image Label:",first_image.img_label)
        print("PCA Image Shape:",first_image.img_pca_approx.shape)
        first_image.display_img(3)

    #Collect the PCA_reduced_features for all images in a matrix
    X = imgCollection.return_PCA_features()
    target = imgCollection.return_data_labels()
    #print(X.shape)
    #print(target.shape)
    #print("Target:",target)

    #print("Reduced Image Features Shape:",X.shape)
    #print("Data Labels Shape:",target.shape)

    imgCollection.acquire_target(target)
    imgCollection.acquire_data(X)
    imgCollection.get_data_and_target()

    #Run the MLP model using the standard test-train split
    #MLP_model = ML_models.MLP_Class(X,target)
    #MLP_model.initialize_MLP()
    #MLP_model.generate_MLP_models()
    #hyperparam_eval_set = (256, 'relu', 0.1, 256, 'relu', 0.05, 256, 'relu', 0.1)
    #MLP_model.evaluate_model(hyperparam_eval_set)

    #Run the MLP model using the SVC for classification
    #SVC_model = ML_models.SVC_Class(X,target)
    #SVC_model.initialize_SVC()
    #SVC_results = SVC_model.run_all_4_SVCs()
    #print(SVC_results)

    #Run the MLP model using leave one out cross validations
    MLP_model = ML_models.MLP_Class(X,target)
    MLP_model.initialize_MLP_LOO()
    hyperparam_eval_set = (256, 'relu', 0.1, 256, 'relu', 0.1, 256, 'relu', 0.2)
    MLP_model.evaluate_model_LOO(hyperparam_eval_set)

    ###########
    ####### Construct and Execute CNN Model #######
##    X_gray = imgCollection.return_cropped_gray_img_list()
##    print(X_gray.shape)
##    X_gray = X_gray[..., np.newaxis]
##    print(X_gray.shape)
##    print(X_gray[0].shape)
##    CNN_model = ML_models.CNN_Class(X_gray,target)
##    CNN_model.initialize_CNN()
##    CNN_model.construct_and_run_CNN_model()



