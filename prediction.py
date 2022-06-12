# Import libraries
from PIL import Image
from io import BytesIO
import numpy as np
from tensorflow.keras.models import load_model


# Decoding image
def read_imagefile(file):
    #decode the image for the model
    bytes_image = Image.open(BytesIO(file))
    return bytes_image

# Prediction function
def image_prediction(my_image):

    # First we load the model saved with Keras model.save()
    loaded_model = load_model('model/vgg19_model_v1.h5')

    # We convert the image into numpy array and resize it to the (48,48) dimension
    image = np.asarray(my_image.resize((48, 48)))[..., :3]

    # Add a dimension to the array to put it into the right format
    x_resize = np.expand_dims(image,axis=0)

    # Make the prediction
    img_pred=np.argmax(loaded_model.predict(x_resize), axis=1)

    # Create a dict with the keys and values
    dict_img = {
        0 : "Airplane",
        1 : "Automobile",
        2 : "Bird",
        3 : "Cat",
        4 : "Deer",
        5 : "Dog",
        6 : "Frog",
        7 : "Horse",
        8 : "Ship",
        9 : "Truck"
    }

    # Find the label of the prediction
    result_pred =  [value for key, value in dict_img.items() if img_pred == key][0]
        
    #return the predicted label
    return result_pred
