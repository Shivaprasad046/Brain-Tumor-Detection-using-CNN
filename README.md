# Brain-Tumor-Detection-using-CNN
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import os
import keras 
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('dark_background')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder 
#
datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
#Loading image
pic = load_img('/image.jpg')
pic_array = img_to_array(pic)
pic_array.shape
pic_array = pic_array.reshape((1,) + pic_array.shape) 
pic_array.shape
count = 0
for batch in datagen.flow(pic_array, batch_size=5,save_to_dir="/Untitled Folder", save_prefix='image', save_format='jpeg'):
    count += 1
    if count > 100:
        break
encoder = OneHotEncoder()
encoder.fit([[0], [1]])
data = []
paths = []
result = []

for r, d, f in os.walk(r'/Microbleed Folder'):
    for file in f:
        if '.jpeg' in file:
            paths.append(os.path.join(r, file))

for path in paths:
    img = Image.open(path)
    img = img.resize((128,128))
    img = np.array(img)
    if(img.shape == (128,128,3)):
        data.append(np.array(img))
        result.append(encoder.transform([[0]]).toarray())

paths = []
for r, d, f in os.walk(r'/Non Microbleed Folder'):
    for file in f:
        if '.jpg' in file:
            paths.append(os.path.join(r, file))

for path in paths:
    img = Image.open(path)
    img = img.resize((128,128))
    img = np.array(img)
    if(img.shape == (128,128,3)):
        data.append(np.array(img))
        result.append(encoder.transform([[1]]).toarray())
data = np.array(data)
data.shape
result = np.array(result)
result = result.reshape(219,2)
x_train,x_test,y_train,y_test = train_test_split(data, result, test_size=0.2, shuffle=True, random_state=0)
model = Sequential()

model.add(Conv2D(32, kernel_size=(2, 2), input_shape=(128, 128, 3), padding = 'Same'))
model.add(Conv2D(32, kernel_size=(2, 2),  activation ='relu', padding = 'Same'))


model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, kernel_size = (2,2), activation ='relu', padding = 'Same'))
model.add(Conv2D(64, kernel_size = (2,2), activation ='relu', padding = 'Same'))

model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

model.compile(loss = "categorical_crossentropy", optimizer='Adamax')
print(model.summary())
y_train.shape
history = model.fit(x_train, y_train, epochs = 30, batch_size = 40, verbose = 1,validation_data = (x_test, y_test))
score=model.evaluate(x_test,y_test,verbose=0)
#Ploting Graph
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Test', 'Validation'], loc='upper right')
plt.show()
#
def names(number):
    if number==0:
        return 'there is a microbleed'
    else:
        return 'there is no microbleed'  
from matplotlib.pyplot import imshow
img = Image.open(r"/Microbleed Folder/image_0_1090.jpeg")
x = np.array(img.resize((128,128)))
x = x.reshape(1,128,128,3)
res = model.predict_on_batch(x)
classification = np.where(res == np.amax(res))[1][0]
imshow(img)
print(str(res[0][classification]*100) + '% accuracy that ' + names(classification))
#
from matplotlib.pyplot import imshow
img = Image.open(r"/image1.jpg")
x = np.array(img.resize((128,128)))
x = x.reshape(1,128,128,3)
res = model.predict_on_batch(x)
classification = np.where(res == np.amax(res))[1][0]
imshow(img)
print(str(res[0][classification]*100) + '% accuracy that ' + names(classification))
#
from google.colab import files

drive.mount()
files
!zip -r "/Microbleed Folder.zip" "/Microbleed Folder" 
from google.colab import files
files.download("/Microbleed Folder.zip")
