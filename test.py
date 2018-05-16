from keras.models import load_model
import cv2
import numpy as np 
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import pandas as pd 
img = cv2.imread('strawberry.jpg')
img = cv2.resize(img,(100,100))
img = np.array(img)
img = img.astype('float')
img /= 255
#img = np.expand_dims(img,axis=3)
img = np.expand_dims(img,axis=0)


print(img.shape)
model = load_model('model/fruitnet.hdf5')


label = pd.read_csv('labels.csv',sep=',')
y = np.array(label['fruit_name'])
label_encoder = LabelEncoder()
integer_labels = label_encoder.fit_transform(y)
integer_labels = integer_labels.reshape(len(integer_labels), 1)  



x = label_encoder.inverse_transform(model.predict_classes((img)))
print(x)
if x == 0:
	print('AppleGolden  ',x)
elif x ==1:
	print('AppleRed')
elif x ==2:
	print('Apricot')
elif x ==3:
	print('Avocado',x)
elif x ==4:
	print('Banana')
elif x ==5:
	print('cherry')
elif x ==6:
	print('coconut')
elif x ==7:
	print('Dates')
elif x ==8:
	print('Grapes')
elif x ==9:
	print('Kiwi')
elif x ==10:
	print('lemon')
elif x ==11:
	print('litchi')
elif x ==12:
	print('mango')
elif x ==13:
	print('orange')
elif x ==14:
	print('peach')
elif x ==15:
	print('pear')
elif x ==16:
	print('pineapple')
elif x ==17:
	print('plum')
elif x ==18:
	print('pomegranate')
elif x ==19:
	print('strawberry')
