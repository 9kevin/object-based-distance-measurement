
# Importing main libraries
import threading
import time

import numpy as np
import os
import six.moves.urllib as urllib
import tarfile
import tensorflow as tf
import cv2

import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

# Use a service account
cred = credentials.Certificate('sec/kevin-detect-distance-2112ad503e11.json')
firebase_admin.initialize_app(cred)

db = firestore.client()

# Importing modules of the object detection
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# Create an Event for notifying main thread.
callback_done = threading.Event()

doc_ref = db.collection(u'cotroller').document(u'main')
# docs = cont_colletion.stream()

controller = True
period = 0
status = False


# Create a callback on_snapshot function to capture changes
def on_snapshot(doc_snapshot, changes, read_time):
    global controller
    global status
    status = doc_snapshot[0].to_dict()['status']
    controller = doc_snapshot[0].to_dict()['control']
    print(controller)
    
    callback_done.set()
    



# Watch the document
doc_ref.on_snapshot(on_snapshot)



def session_over():
    global controller
    time.sleep(30)
    controller = False

    # patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

    # Patch the location of gfile
tf.gfile = tf.io.gfile

    # Model preparation 
    # What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
# MODEL_FILE = MODEL_NAME + '.tar.gz'
    # DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

    # Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

    # List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90
        
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.compat.v1.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')
        
    # List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = 'models/research/object_detection/data/mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
        
    # Importing the video file
filename = "sample_video.mp4"


# Reading the video with open cv
def start_detection():
    global controller
    global period
    global doc_ref

    cap = cv2.VideoCapture(0)

    with detection_graph.as_default():
        with tf.compat.v1.Session(graph = detection_graph) as sess:
            while True:

                
                ret, image_np = cap.read()
                    # Expand dimensions
                image_np_expanded = np.expand_dims(image_np, axis=0)
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                    # Creating boxes to highlight a given object
                boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                scores = detection_graph.get_tensor_by_name('detection_scores:0')
                classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                    # Actual detections
                boxes, scores, classes, num_detections = sess.run(
                    [boxes, scores, classes, num_detections], 
                    feed_dict = {image_tensor: image_np_expanded}
                )
                    # Visualization of the results
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np, 
                    np.squeeze(boxes), 
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8
                )
                
                # Measuring the distance
                for i,b in enumerate(boxes[0]):
                    if classes[0][i] == 1:
                            
                        if scores[0][i] >= 0.5:

                            mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
                            mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
                            apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1]))**4),1)
                            cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x*800),int(mid_y*450)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)                  
                            if apx_distance < 0.1:
                                print('........++   ' + str(controller))
                                        
                                if mid_x > 0.3 and mid_x < 0.7:

                                            # Displaying a warning message
                                            
                                    print('>>>>>>>>>>>>>>>>>>>>  WARNING  <<<<<<<<<<<<<<<<<<<<<<<')
                                            # controller = False
                                    # cv2.putText(image_np, 'WARNING!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                                    print('Warned the user')
                                    # update the status 
                                    if not status:
                                        doc_ref.update({
                                            'status' : True
                                        })

                                # controller = False

                
            # cv2.imshow("object detection", cv2.resize(image_np, (600, 800)))
                if not controller:
                        # cv2.destroyAllWindows()
                    #turn status off
                    doc_ref.update({
                        'status' : False
                    })

                    return 'Done'
                      
                
                if period == 29:

                    # status off
                    doc_ref.update({
                        'status' : False
                    })
                    return 'timeout'
                        
                            
                            
                            
                period = period + 1
                print(period)
                time.sleep(1)
                            
                
                    
           

# Detect.detect_app()