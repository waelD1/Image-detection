# FastAPI libraries
from fastapi import FastAPI, Request, Form, File, UploadFile
from fastapi.templating import Jinja2Templates
from fastapi.responses import Response

# Tensorflow libraries
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np

# Import function from prediction.py file
from prediction import read_imagefile, image_prediction

# Lauching the app
app = FastAPI()
templates = Jinja2Templates(directory="./website")

# Set up the home page
@app.get("/")
async def read_items(request : Request):
    return templates.TemplateResponse('base.html', {"request" : request})

#Create the prediction
@app.post("/result")
async def create_upload_file(request : Request, upload_image: UploadFile = File(...)):

    # Read the uploaded image
    contents = await upload_image.read()

    # Decode the image with the function that is in the prediction.py file
    my_image = read_imagefile(contents)

    # Predict the label of the image
    result_pred = image_prediction(my_image)


    return templates.TemplateResponse("result.html", {'request' : request, 'prediction' : result_pred })