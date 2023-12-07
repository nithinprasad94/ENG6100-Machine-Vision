# Project:
# Authors: Helena Kunic, Nithin Prasad

import cv2
import preprocessing
import ML_models

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

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

    pca_vals = [19]
    for i in pca_vals:
        imgCollection.apply_PCA_on_all_images(i) #NUMBER OF PCA COMPONENTS: start at 19 components for 95% loss ... might still be missing defects
        #first_image = imgObjList[0]
        #print("First Image Label:",first_image.img_label)
        #print("PCA Image Shape:",first_image.img_pca_approx.shape)
        #first_image.display_img(3)

    #Collect the PCA_reduced_features for all images in a matrix
    X = imgCollection.return_PCA_features()
    target = imgCollection.return_data_labels()
    print("Target:",target)

    print("Reduced Image Features Shape:",X.shape)
    print("Data Labels Shape:",target.shape)

    imgCollection.acquire_target(target)
    imgCollection.acquire_data(X)
    imgCollection.get_data_and_target()

    MLP_model = ML_models.MLP_Class(X,target)
    MLP_model.initialize_MLP()
    #MLP_model.generate_MLP_models()
    MLP_model.evaluate_model()

    SVC_model = ML_models.SVC_Class(X,target)
    SVC_model.initialize_SVC()
    SVC_results = SVC_model.run_all_4_SVCs()
    print(SVC_results)

