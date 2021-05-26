from flask import Flask
from flask import render_template
from flask import request
from flask import jsonify
from PIL import Image
import base64
import cv2
from base64 import decodestring
import io


app = Flask(__name__)


@app.route('/')
def hello():
    return render_template('index.html')

@app.route('/what', methods=['GET', 'POST'])
def image():
    imdata = request.get_json()['value']
    imdata = imdata[imdata.find('/9'):]
    image = Image.open(io.BytesIO(base64.b64decode(imdata))).save('result.jpg')
    x = cv2.imread('result.jpg')[:,:,::-1].mean()
    #return render_template('index.html')
    return jsonify({'label': str(x)})
