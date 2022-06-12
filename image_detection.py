# import the libraries as shown below

from tensorflow.keras import datasets, optimizers, losses, callbacks
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten,Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import MaxPooling2D
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing import image
# One hot label encoding
from tensorflow.keras.utils import to_categorical


# Load the dataset and print the shapes
(X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()
assert X_train.shape == (50000, 32, 32, 3)
assert X_test.shape == (10000, 32, 32, 3)
assert y_train.shape == (50000, 1)
assert y_test.shape == (10000, 1)

print("Shape of x_train is ",X_train.shape)
print("Shape of y_train is ",y_train.shape)
print("Shape of x_test  is ",X_test.shape)
print("Shape of y_test  is",y_test.shape)

# Resize function to get a 48 by 48 image dimension
def resize_img(img):
    numberOfImage = img.shape[0]
    new_array = np.zeros((numberOfImage, 48,48,3))
    for i in range(numberOfImage):
        new_array[i] = cv2.resize(img[i,:,:,:],(48,48))
    return new_array

# Resize the training set
X_train = resize_img(X_train)
X_test = resize_img(X_test)
print("New shape of x_train is ",X_train.shape)
print("New shape of x_test  is ",X_test.shape)

# One hot encoding of the target features
y_train = to_categorical(y_train,num_classes=10)
y_test = to_categorical(y_test,num_classes=10)

print("New shape of y_train is ",y_train.shape)
print("New shape of y_test  is ",y_test.shape)

# VGG19 model
# Add fully connected layers to layer and use pretrained weights
vgg = VGG19(include_top=False,weights="imagenet",input_shape=(48,48,3))
vgg.summary()


# Creating the layers to add to the Vgg19 model
model = Sequential()

# Adding layers to the blank model
for layer in vgg.layers:
    model.add(layer)
    
# Don't train layers again, because they are already trained
for layer in model.layers:
    layer.trainable = False
    
# Adding fully connected layers
model.add(Flatten())
model.add(Dense(416))
model.add(Dense(10,activation="softmax"))

# summary of the model
model.summary()

#Compiling the model
opt = optimizers.Adam(learning_rate=0.0001)
model.compile(loss="categorical_crossentropy",metrics=["accuracy"], optimizer=opt)

# Training of the model
hist = model.fit(X_train,y_train,validation_split=0.15,epochs=20,batch_size=1000)


# Loss Graph
plt.subplots(figsize=(10, 6))
plt.plot(hist.epoch,hist.history["loss"],color="green",label="Train Loss")
plt.plot(hist.epoch,hist.history["val_loss"],color="blue",label="Validation Loss")
plt.xlabel("Epoch Number")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss Graph")
plt.show()

# Accuracy Graph
plt.subplots(figsize=(10,6))
plt.plot(hist.epoch,hist.history["accuracy"],color="green",label="Train Accuracy")
plt.plot(hist.epoch,hist.history["val_accuracy"],color="blue",label="Validation Accuracy")
plt.xlabel("Epoch Number")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Accuracy Graph")
plt.show()


# Prediction of the model
y_pred = model.predict(X_test)
y_pred = np.round(y_pred).astype(int)

#Classification metrics and evluation of the model
print(classification_report(y_test, y_pred))
print((model.evaluate(X_test, y_test, return_dict=True)))

# Saving the model
# model.save("vgg19_model_v1.h5")


# Make a prediction
# Load the image
img=image.load_img('my_image.jpg',target_size=(48,48))

# Convert it into numpy array
x= image.img_to_array(img)

# Add a dimension to the array to put it into the right format
x_resize = np.expand_dims(x,axis=0)

# Make the prediction
img_pred=np.argmax(model.predict(x_resize), axis=1)

# Print the result
if(img_pred==0):
    print("Airplane")
elif(img_pred ==1):
    print("Automobile")
elif(img_pred ==2):
    print("Bird")
elif(img_pred ==3):
    print("Cat")
elif(img_pred ==4):
    print("Deer")
elif(img_pred ==5):
    print("Dog")
elif(img_pred ==6):
    print("Frog")
elif(img_pred ==7):
    print("Horse")
elif(img_pred ==8):
    print("Ship")
elif(img_pred ==9):
    print("Truck")








