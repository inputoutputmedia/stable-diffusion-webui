import os
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig, LlavaNextProcessor, LlavaNextForConditionalGeneration, VipLlavaForConditionalGeneration
import transformers
import time
from datetime import datetime
import numpy as np
import cv2
from ultralytics import YOLO
import json
from typing import Union, List
from PIL import Image
import requests
from io import BytesIO
import torch
import gc
import re
import logging
from pathlib import Path
import sys
import sqlite3

def predict(chosen_model, img, classes=[], conf=0.5):
    if classes:
        results = chosen_model.predict(img, classes=classes, conf=conf)
    else:
        results = chosen_model.predict(img, conf=conf)

    return results

def predict_and_detect(chosen_model, img, classes=[], conf=0.5):
    results = predict(chosen_model, img, classes, conf=conf)
    boxes = []
    names = []
    confs = []

    for box in results[0].boxes:

        boxes.append((int(box.xyxy[0][0]), int(box.xyxy[0][1]), int(box.xyxy[0][2]), int(box.xyxy[0][3])))

        names.append(results[0].names[int(box.cls[0])])

        confs.append(float(box.conf[0]))

    return boxes, names, confs

def load_image(image_path_or_url: str) -> Image.Image:
    if image_path_or_url.startswith(('http://', 'https://')):
        response = requests.get(image_path_or_url)
        image = Image.open(BytesIO(response.content))
    else:
        image = Image.open(image_path_or_url)
    return image   

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

currentDir = str(Path(os.getcwd()).absolute())

# with open(currentDir+"/mysql_connection_input.txt", 'r') as file:
#     lines = [line.rstrip() for line in file]

# mydb = mysql.connector.connect(
#     host=lines[2].split('=')[1],
#     user=lines[3].split('=')[1],
#     password=lines[4].split('=')[1],
#     database=lines[5].split('=')[1])

assetsDir = "C:/decentre/appdata/assets"

isExist = os.path.exists(assetsDir)
if not isExist:
    # Create a new directory because it does not exist
    os.makedirs(assetsDir)

# mycursor = mydb.cursor()

settingsFile = currentDir+"/extensions/decentre/settings.st"

file = open(settingsFile, 'r')

objWidth = int(file.readline()[18:])
objHeight = int(file.readline()[19:])
confl = float(file.readline()[22:])
capLen = int(file.readline()[20:])

file.close()

dbFile = "C:/decentre/appdata/decentre.db"
db = sqlite3.connect(dbFile)
mycursor = db.cursor()

def detect_image(image_path_or_url: str, isCaption: str, model_path: str, maxImgId : int):
    sql2 = "SELECT IFNULL(MAX(object_id), 0) FROM object_detection_table"
    sql3 = "INSERT INTO image_table (image_id, image_path, prompt_text, generation_time, user_id) VALUES (?, ?, ?, ?, ?)"
    sql4 = "INSERT INTO object_detection_table (object_id, image_id, model_used, object_name, object_path, confidence_score, bounding_box, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)"

    mycursor.execute(sql2)
    maxObjId = mycursor.fetchone()[0] + 1
  
    # if image_path_or_url.startswith(('http://', 'https://')):
    #     resp = urlopen(image_path_or_url)
    #     image = np.asarray(bytearray(resp.read()), dtype="uint8")
    #     image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    # else:
    model = YOLO("C:/decentre/appdata"+"/models/detection/"+model_path)
    image = cv2.imread(image_path_or_url)
    boxes,names, confs = predict_and_detect(model, image, classes=[], conf=confl)
    # image_id, image_path, prompt_text, generation_time, user_id
    created_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    val1 = (maxImgId, image_path_or_url, "", created_at, 4)
    mycursor.execute(sql3, val1)
    db.commit()

    # object_id, image_id, model_used, object_name, object_path, confidence_score, bounding_box, created_at
    val2 = []
    #print("obj detection:{}".format(len(boxes)))
    for i in range(0, len(boxes)):
        width = boxes[i][2] - boxes[i][0]
        height = boxes[i][3] - boxes[i][1]
        if(width < objWidth or height < objHeight):
            continue

        box = json.dumps({"x":boxes[i][0], "y":boxes[i][1], "width":width, "height":height})
        val2.append((maxObjId, maxImgId, model_path, names[i], assetsDir+'/{img_id}-{obj_id}.jpg'.format(img_id = maxImgId, obj_id = maxObjId), confs[i], box,   created_at))
        obj = image[boxes[i][1]:boxes[i][3], boxes[i][0]:boxes[i][2]]
        cv2.imwrite(assetsDir+'/{img_id}-{obj_id}.jpg'.format(img_id = maxImgId, obj_id = maxObjId), obj)
        maxObjId += 1

    mycursor.executemany(sql4, val2)
    db.commit()

    del model
    gc.collect()
    torch.cuda.empty_cache()        

    print("objs detected")
    db.close()    

detect_image(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])) 