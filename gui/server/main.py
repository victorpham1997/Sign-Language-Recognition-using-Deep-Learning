from flask import Flask, request
from flask_cors import CORS, cross_origin
import time
import os
import ffmpeg
from model import *
import torch

device = torch.device('cpu')
model = torch.load('./modelweights-baseline.pt', map_location=device)
model.eval()

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


import cv2

@app.route('/identify-sign', methods=['POST'])
@cross_origin()
def identify_sign():
    #1. we save the file that was sent to us
    video = request.files['video']
    filename = f'{time.time()}'
    video.save(filename + '.webm')    

    #2. we convert it from webm to mp4
    stream = ffmpeg.input(filename + '.webm')
    stream = ffmpeg.filter(stream, 'fps', fps=30, round='up').filter('pad', 'in_w', 512).filter('crop', 512, 512)
    stream = ffmpeg.output(stream, filename + '.mp4')
    ffmpeg.run(stream)
    os.remove(filename + '.webm') #clean up

    #3. we send it to the model for prediction
    prediction = video_to_word(filename + '.mp4', model)
    print(prediction)
    return prediction

Flask.run(app)

# export FLASK_APP=main.py
# python3 -m flask run