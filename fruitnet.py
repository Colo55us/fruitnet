import pandas as pd 
import numpy as np 
from os import listdir
import cv2

from sklearn import model_selection,utils
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout,normalization,Activation
from keras.optimizers import Adam
from keras import callbacks
from keras import backend as K

K.set_image_dim_ordering('tf')
def encode_labels(y):
    label_encoder = LabelEncoder()
    integer_labels = label_encoder.fit_transform(y)
    one_hot = OneHotEncoder(sparse = False)
    integer_labels = integer_labels.reshape(len(integer_labels), 1)    
    y = one_hot.fit_transform(integer_labels)
    return y

if __name__=='__main__':
    '''
    path_from = 'images/dump/'

    li_imgs = listdir(path_from)
    img_array = []

    for each_img in li_imgs:
        img = cv2.imread(path_from+each_img)
        img_array.append(img)

    img_array = np.array(img_array).astype('float32')
    img_array /= 255
    np.save('img_array.npy',img_array)
    '''
    #print(img_array.shape,'                 ',img_array[0].shape)

    img_array = np.load('img_array.npy')
    rows = 100
    cols = 100
    n_epochs = 10
    n_classes = 20
    label = pd.read_csv('labels.csv',sep=',')
    y = np.array(label['fruit_name'])
    y = encode_labels(y)

    X,y = utils.shuffle(img_array,y)

    X_train,X_test,y_train,y_test = model_selection.train_test_split(X,y,test_size=0.15,random_state = 42)
    
    input_shape = img_array[0].shape

    model = Sequential()
    model.add(Conv2D(96,(11,11),strides=4,padding='valid',input_shape=input_shape,activation='relu'))
    model.add(MaxPooling2D(3,strides=2))
    model.add(normalization.BatchNormalization())

    model.add(Conv2D(256,(11,11),padding='same',activation='relu'))
    model.add(MaxPooling2D(3,strides=2))
    model.add(normalization.BatchNormalization())

    model.add(Conv2D(384,(3,3),padding='same',activation='relu'))
    model.add(Conv2D(384,(3,3),padding='same',activation='relu'))
    model.add(Conv2D(256,(3,3),padding='same',activation='relu'))

    model.add(MaxPooling2D(3,strides=2))
    model.add(normalization.BatchNormalization())
    model.add(Flatten())
    model.add(Dense(4096))
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))

    model.add(Dense(4096))
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))

    model.add(Dense(n_classes))
    model.add(Activation('softmax'))

    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

    tb = callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0,write_graph=True, write_images=True)

    model.fit(X_train,y_train,batch_size = 25,epochs=n_epochs,verbose=1,validation_split=0.2,class_weight='auto',callbacks=[tb])

    score = model.evaluate(X_test,y_test,verbose=1)
    print('Model score- ',score[0])
    print('Model accuracy- ',score[1])

    model.save('model/fruitnet.hdf5')






