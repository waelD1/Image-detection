from fastapi import FastAPI, Request, Form, File, UploadFile
from fastapi.templating import Jinja2Templates
from fastapi.responses import Response

from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
from random import randint

from prediction import read_imagefile, image_prediction


app = FastAPI()
templates = Jinja2Templates(directory="./website")



# Set up the home page
@app.get("/")
async def read_items(request : Request):
    return templates.TemplateResponse('base.html', {"request" : request})

#Create the prediction
@app.post("/result")
async def create_upload_file(request : Request, upload_image: UploadFile = File(...)):

    # read the uploaded image
    contents = await upload_image.read()

    # decode the image with the function that is in the prediction.py file
    my_image = read_imagefile(contents)

    # predict the label of the image
    result_pred = image_prediction(my_image)


    return templates.TemplateResponse("result.html", {'request' : request, 'prediction' : result_pred })

