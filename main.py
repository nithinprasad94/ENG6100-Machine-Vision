# Project:
# Authors: Helena Kunic, Nithin Prasad

import cv2
import preprocessing

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    fileObj = preprocessing.File_Reader("training_images")
    imgCollection = fileObj.get_fnames_from_dir()
    imgCollection.crop_all_imgs(450, 1750, 200, 1454)
    imgObjList = imgCollection.img_obj_list
    print(len(imgObjList))
    first_image = imgObjList[0]
    first_image.display_img(0)
    first_image.display_img(1)
    imgCollection.PCA_for_crop_imgs(2)
