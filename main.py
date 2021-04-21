from flask import Flask


app = Flask(__name__)


# Importing main libraries
import numpy as np
import os
import six.moves.urllib as urllib
import tarfile
import tensorflow as tf
import cv2

# Importing modules of the object detection
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

import object_detection1



# create the index route
@app.route("/", methods= ['GET'])
def index():
    return "Welcome to Ksave"

@app.route("/detect", methods= ['GET'])
def detect():
    

    if object_detection1.controller:
        #run model
        return object_detection.start_detection()
        
    else:
        # stopped
        return 'Start'

#run the app
if __name__ == "__main__":
    # host="127.0.0.1", port=8080
    app.run(debug=True)