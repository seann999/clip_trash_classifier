from flask import Flask
from flask import render_template
from flask import request
from flask import jsonify
from PIL import Image
import base64
import numpy as np
import io
from classifier import Classifier


app = Flask(__name__)
model = Classifier()


@app.route('/')
def hello():
    return render_template('index.html')


@app.route('/classify', methods=['GET', 'POST'])
def image():
    imdata = request.get_json()['value']
    imdata = imdata[imdata.find('/9'):]
    x = np.array(Image.open(io.BytesIO(base64.b64decode(imdata))))
    cat, obj = model.classify(x)

    return jsonify({'label': cat, 'msg': obj})
