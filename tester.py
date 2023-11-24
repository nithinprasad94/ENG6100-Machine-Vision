import os
import cv2
import matplotlib

fpath = "./training_images/150924030660/1_0.png"
orig_img = cv2.imread(fpath)
print(orig_img.shape)
cropped_img = orig_img[:,:]
cropped_img = orig_img[450:1750,200:]
#print(cropped_img.shape)
#cv2.namedWindow('Image',cv2.WINDOW_NORMAL)
#cv2.imshow("Cropped Image:",cropped_img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

x0 = orig_img.shape[0]
y0 = orig_img.shape[1]
x1 = cropped_img.shape[0]
y1 = cropped_img.shape[1]
efficiency = ((x1*y1)/(x0*y0))*100
print("Crop Efficiency:",efficiency)

import matplotlib.pyplot as plt

image = cv2.imread('img.jpg')
color_image = cv2.cvtColor(cropped_img,cv2.COLOR_BGR2RGB)

plt.imshow(color_image)
plt.axis('off')
plt.show()

