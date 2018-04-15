import numpy as np
import six.moves.urllib as urllib
import sys, os, math
from .memoryCache import memoryCache
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image, ImageGrab

import cv2
import keyboard
global lastKey
lastKey = None
def on_press(key):
    global lastKey
    lastKey = key.name
keyboard.hook(on_press)
#cap = cv2.VideoCapture(0)

# from color_extractor import ImageToColor

# npz = np.load('color_names.npz')
# img_to_color = ImageToColor(npz['samples'], npz['labels'])

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")


# ## Object detection imports
# Here are the imports from the object detection module.

# In[3]:

from utils import label_map_util

from utils import visualization_utils as vis_util


# # Model preparation

# ## Variables
#
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.
#
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.

# In[4]:

# What model to download.
MODEL_NAME =  'ssd_mobilenet_v1_coco_2017_11_17'#'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90
tolerance = 0.15


# ## Download Model

# In[5]:
print('now loading graph from file')
# opener = urllib.request.URLopener()
# opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
# tar_file = tarfile.open(MODEL_FILE)
# for file in tar_file.getmembers():
#   file_name = os.path.basename(file.name)
#   if 'frozen_inference_graph.pb' in file_name:
#     tar_file.extract(file, os.getcwd())


# ## Load a (frozen) Tensorflow model into memory.

# In[6]:
print('now loading frozen graph into model')
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

# In[7]:

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# ## Helper code

# In[8]:

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


# # Detection

# In[9]:

# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = 'test_images'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3) ]

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

# return centre position of box with coords: x1,y1,x2,y2
def getBoxCentre(box):
    return (box[0]+box[2])/2, (box[1]+box[3])/2

opers = detection_graph.get_operations()
# for i in opers: print(i)
# In[10]
#prevdata stores previous frame details
# oldBoxes = None
# oldFeatures = None
# nettotal = 0.0
cache = memoryCache()
commands = ['']
validIndex = 0
validKeys = ['living room','kitchen','dining room','veranhdah'  ,'stairs']
training = False
running = False
prevKey = None
print('running predictor')
with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    while True:
    #   ret, image_np = cap.read()
      image_np = np.array(ImageGrab.grab())
      image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
      hsplit = np.split(image_np, 2)
      vsplit = np.split(np.array(hsplit), 2, axis=2)
      segments = np.concatenate(vsplit,axis=0)
      resized = cv2.resize(image_np, (0,0), fx=0.5, fy=0.5)
      segments = np.concatenate([segments, np.expand_dims(resized,axis=0)],axis=0) # add main image to process
      # print(detection_graph.get_operations())
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    #   image_np_expanded = np.expand_dims(image_np, axis=0)
      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      # Each box represents a part of the image where a particular object was detected.
      boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      # Each score represent how level of confidence for each of the objects.
      # Score is shown on the result image, togeth  er with the class label.
      scores = detection_graph.get_tensor_by_name('detection_scores:0')
      classes = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')
      # Actual detection.
      (boxes, scores, classes, num_detections) = sess.run(
          [boxes, scores, classes, num_detections],
          feed_dict={image_tensor: segments})
      tolerance = np.partition(scores, -5,axis=1)[:,-5]
      # print(boxes)
      # Visualization of the results of a detection.
      # insert into memory correlation of vision to room
      boxes[0:4] = boxes[0:4]/2
      boxes[1,:,(0,2)] += 0.5
      boxes[2,:,(1,3)] += 0.5 # repositions quarters
      boxes[3,:,:] += 0.5 # repositions quarters
      indexes = scores > np.tile(np.array([tolerance]).transpose(), (1, 100)) # boolean matrix of if features are confidently enough observed
      features = classes[indexes].flatten() # high confidence featurees
      goodBoxes = boxes[indexes].reshape(-1,4)
      goodScores = scores[indexes].flatten()
      centres = (goodBoxes[:,(0,1)]+goodBoxes[:,(2,3)])/2
    
      #get color features
    #   colors = np.empty([nboxes.shape[0]],dtype=object)
    #   for i in range(len(nscores)):
        #   if nscores[i] >= tolerance:
        #       if i < 0:
        #         topbox = nboxes[i]
        #         # sh = image_np.shape
        #         # x1 = int(topbox[0]*sh[0])
        #         # y1 = int(topbox[1]*sh[1])
        #         # x2 = int(topbox[2]*sh[0])
        #         # y2 = int(topbox[3]*sh[1])
        #         # img2 = image_np[x1:x2,y1:y2]
        #         # colors[i] = img_to_color.get(img2)[0]
        #       else:
        #         colors[i] = '' 
      #get shape feature
      concepts = list(map(str, list(features)))
      newHashes = []
      for m in [1,2,4,8,16,32]:
        quadrants = (goodBoxes*m).astype(int)
        for i in range(len(concepts)):
            newHashes.append(concepts[i] +',m:'+str(m)+',x:'+ str(quadrants[i][0]) +',y:'+ str(quadrants[i][1])) 
            newHashes.append(concepts[i] +',m:'+str(m)+',x:'+ str(quadrants[i][0]) )
            newHashes.append(concepts[i] +',m:'+str(m)+',y:'+ str(quadrants[i][1]) )
            for j in range(len(concepts)):
                 newHashes.append(concepts[i] +','+concepts[j]+',m:'+str(m)+',x:'+ str(quadrants[i][0]-quadrants[j][0]) +',y:'+ str(quadrants[i][1]-quadrants[j][1])) 
      concepts = (newHashes + commands)

      if training: cache.reinforce(concepts,label=commands[0])
      cache.propogate(concepts)
      pred, score = cache.getHighest(validKeys)
      print('prediction:'+pred, ' confidence:'+str(score))
    #   if running:
    #     if pred != 'stationary':
    #       keyboard.press_and_release(pred)
    #       prevKey = pred
        # else:
        #   keyboard.release(prevKey)
        #   prevKey = None
      cache.saveCache()
      ##############

      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(goodBoxes),
          np.squeeze(features).astype(np.int32),
          goodScores,
          category_index,
          use_normalized_coordinates=True,
          min_score_thresh=0.0,
          line_thickness=8,
          max_boxes_to_draw=5*10
          )
      cv2.putText(
          image_np,
          pred, 
          (10,100), 
          cv2.FONT_HERSHEY_SIMPLEX, 
          3,
          (0,0,0),
          4
      )
    #   image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
      cv2.imshow('object detection', cv2.resize(image_np, (800,600)))
      res = cv2.waitKey(25) & 0xFF
      if res == ord('q'):
        cv2.destroyAllWindows()
        break
        # elif res == ord('a'):

        #     # keyboard.press('a')
        #     commands.append( 'square' )
        #     print('square class')
        # elif res == ord('s'):
        #     # keyboard.press('s')
        #     commands.append( 'circle' )
        #     print('circle class')
    #   commands = []
      if lastKey:
        #   if lastKey in validKeys:
        #       commands = [lastKey]
          if lastKey == 'a':
              validIndex -= ( validIndex - 1) % len(validKeys)
              print(validKeys[validIndex])
              commands = [validKeys[validIndex]]
              lastKey = None
          elif lastKey == 's':
              validIndex =  ( validIndex + 1) % len(validKeys)
              print(validKeys[validIndex])
              commands = [validKeys[validIndex]]
              lastKey = None
          elif lastKey == 'x':
              commands = ['']
              lastKey = None

          elif lastKey == 'r':
              training = not training
              print('training?'+str(training))
              lastKey = None
          elif lastKey == 't':
            running = not running
            lastKey = None 
            print('running?'+str(running))
          

keyboard.unhook_all()