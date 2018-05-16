import os 
import cv2
import numpy as np 



def increase_imgs(path):
    img = cv2.imread(path)
    img2 = cv2.flip(img,1)
    img3 = cv2.flip(img,0)
    img4 = cv2.flip(img2,0)

    return img,img2,img3,img4




if __name__=='__main__':
	x = input()
	path_from = 'images/'+str(x)+'/'
	path_to = 'images/'+str(x)+'_final/'
	li_imgs = os.listdir(path_from)
	count=0
	for imgs in li_imgs:
		path = path_from+str(imgs)
		img1,img2,img3,img4 = increase_imgs(path)
		cv2.imwrite(path_to+str(x)+'_1-'+str(count)+'.jpg',img1)
		cv2.imwrite(path_to+str(x)+'_2-'+str(count)+'.jpg',img2)
		cv2.imwrite(path_to+str(x)+'_3-'+str(count)+'.jpg',img3)
		cv2.imwrite(path_to+str(x)+'_4-'+str(count)+'.jpg',img4)
		count +=1 







