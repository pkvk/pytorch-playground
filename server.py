"""

# How to run the server

1) Set virtual environment (optional):

   $ pip3 install virtualenv
   $ virtualenv venv
   $ . venv/bin/activate

2) Install pytorch and install other requirements:

    $ pip3 install gunicorn Flask

3) Run (four) web workers:

    $ CONFIG=config.cfg gunicorn -w 4 server:app

4) Open http://localhost:8000 in your browser.


"""

from flask import Flask, render_template, request
from PIL import Image
import base64
import io
import numpy as np

import sample


app = Flask(__name__)
app.config.from_envvar('CONFIG')

@app.route('/')
def main():
    return render_template('index.html')

@app.route('/check', methods=['POST'])
def check():
    image_b64 = request.values['imageBase64']
    s = image_b64.split(";base64,")[1]  # Everything after ;base64,
    img_data = base64.b64decode(s)
    im = Image.open(io.BytesIO(img_data))
    res = sample.recognize(im)
    return res
