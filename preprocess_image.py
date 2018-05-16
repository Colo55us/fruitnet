import cv2
import os
import numpy as np



path_from = 'images/DB_Mango/'
path_to = 'images/mango_temp/'
li_imgs = os.listdir(path_from)
print(li_imgs[:5])
white = np.asarray([255,255,255])
black = np.asarray([40,40,40])
count = 1

def rotate_images(img,degree):
    
    img = cv2.flip(img,degree)
    return img
    '''
    rows,cols,_ = img.shape

    M = cv2.getRotationMatrix2D((cols,rows),degree,1)
    dst = cv2.warpAffine(img,M,(cols,rows))
    return dst
    '''

for imgs in li_imgs:
    img = cv2.imread(path_from+imgs,cv2.IMREAD_COLOR)
    img2 = np.array(img,copy=True)
    
    (rows,cols,_) = img.shape
    for row in range(rows):
        for col in range(cols):
            pixel = img[row][col]
            if all(pixel<=black):
                img2[row][col] = white
    
    img2 = cv2.resize(img2,(100,100))
    img2 = img2[5:95,15:75]
    img2 = cv2.resize(img2,(100,100))
    img_90 = rotate_images(img2,1)
    img_180 = rotate_images(img2,0)
    img_270 = rotate_images(img_90,0)
    #img_180 = rotate_images(img2,180)
    #img_270 = rotate_images(img2,270)

    cv2.imwrite('images/mango_temp/mango_ {}.jpg'.format(count),img2)
    cv2.imwrite('images/mango_temp/mango_90 {}.jpg'.format(count),img_90)
    cv2.imwrite('images/mango_temp/mango_180 {}.jpg'.format(count),img_180)
    cv2.imwrite('images/mango_temp/mango_270 {}.jpg'.format(count),img_270)
    count += 1