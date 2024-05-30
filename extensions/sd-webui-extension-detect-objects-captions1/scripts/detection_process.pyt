import os
from transformers import AutoProcessor, LlavaForConditionalGeneration
import transformers
import time
from datetime import datetime
import numpy as np
import cv2
from ultralytics import YOLO
import mysql.connector
import json
from typing import Union, List
from PIL import Image
import requests
from io import BytesIO
import torch
import re
import logging
from pathlib import Path


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

def generate_image_caption(image_path_or_url: str, model_id: str = "llava-hf/llava-1.5-7b-hf") -> List[str]:
    try:
        raw_image = load_image(image_path_or_url)
        model = LlavaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True, load_in_4bit=True)
        processor = AutoProcessor.from_pretrained(model_id)

        prompt = "USER: <image>\nWhat is your assessment for this image?\nASSISTANT:"
        inputs = processor(prompt, raw_image, return_tensors='pt').to("cuda" if torch.cuda.is_available() else "cpu", torch.float16)

        output = model.generate(**inputs, max_new_tokens=300, do_sample=False)
        captions_text = processor.decode(output[0][2:], skip_special_tokens=True)

        captions_cleaned = re.sub(r'^.*?ASSISTANT: ', '', captions_text, flags=re.DOTALL)
        sentences = [sentence.strip() for sentence in re.split(r'(?<=[.])\s+', captions_cleaned) if sentence]

        safe_texts = []
        for sentence in sentences:
            safe_text = sentence.replace("'", "''")
            safe_texts.append(safe_text)

        return safe_texts
    except Exception as e:
        logger.error(f"Error generating caption: {e}")
        raise

print("Hello detection started")

currentDir = str(Path(os.getcwd()).absolute())
print(currentDir)

path_to_watch3 = currentDir+"/output/extras-images"
model = YOLO("C:/Users/User/Desktop/Models/yolov3u.pt")

mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="1*AllahAkbarMJ",
    database="sdwebui_extention_local")

mycursor = mydb.cursor()

sql1 = "SELECT IFNULL(MAX(image_id), 0) FROM image_table"
sql2 = "SELECT IFNULL(MAX(object_id), 0) FROM object_detection_table"
sql3 = "INSERT INTO image_table (image_id, image_path, prompt_text, generation_time, user_id) VALUES (%s, %s, %s, %s, %s)"
sql4 = "INSERT INTO object_detection_table (object_id, image_id, model_used, object_name, object_path, confidence_score, bounding_box, created_at) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
sql5 = "SELECT IFNULL(MAX(caption_id), 0) FROM captions_table"
sql6 = "INSERT INTO captions_table (caption_id, image_id, text, model_used, created_at) VALUES (%s, %s, %s, %s, %s)"

mycursor.execute(sql1)
maxImgId = mycursor.fetchone()[0] + 1
print("Max imgId: ", maxImgId)

mycursor.execute(sql2)
maxObjId = mycursor.fetchone()[0] + 1
print("Max objId: ", maxObjId)

mycursor.execute(sql5)
maxCapId = mycursor.fetchone()[0] + 1
print("Max capId: ", maxCapId)

print("torch.cuda.is_available: ", torch.cuda.is_available())
print("transformers.is_bitsandbytes_available: ", transformers.is_bitsandbytes_available())

isExist = os.path.exists(path_to_watch3)
if not isExist:
    # Create a new directory because it does not exist
    os.makedirs(path_to_watch3) 

assetsDir = currentDir+"/output/assets"

isExist = os.path.exists(assetsDir)
if not isExist:
    # Create a new directory because it does not exist
    os.makedirs(assetsDir)

now = datetime.now()
    
formatted_date = now.strftime('%Y-%m-%d')

path_to_watch1 = currentDir+"/output/txt2img-images/"+formatted_date
path_to_watch2 = currentDir+"/output/img2img-images/"+formatted_date

isExist = os.path.exists(path_to_watch1)
if not isExist:
    # Create a new directory because it does not exist
        os.makedirs(path_to_watch1)

isExist = os.path.exists(path_to_watch2)
if not isExist:
    # Create a new directory because it does not exist
    os.makedirs(path_to_watch2)    
    

before1 = dict ([(f, None) for f in os.listdir (path_to_watch1)])
before2 = dict ([(f, None) for f in os.listdir (path_to_watch2)])
before3 = dict ([(f, None) for f in os.listdir (path_to_watch3)])                  

while True:

    time.sleep (10)

    after1 = dict ([(f, None) for f in os.listdir (path_to_watch1)])
    after2 = dict ([(f, None) for f in os.listdir (path_to_watch2)])
    after3 = dict ([(f, None) for f in os.listdir (path_to_watch3)])
    
    added1 = [f for f in after1 if not f in before1]
    # removed1 = [f for f in before1 if not f in after1]
    
    if added1: 
        print("Added: ", ", ".join(added1))
        for f in added1:
            image = cv2.imread(path_to_watch1+"/"+f)
            boxes,names, confs = predict_and_detect(model, image, classes=[], conf=0.25)
            # image_id, image_path, prompt_text, generation_time, user_id
            created_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            val1 = (maxImgId, path_to_watch1+"/"+f, "", created_at, 4)
            mycursor.execute(sql3, val1)
            mydb.commit()
            
            # object_id, image_id, model_used, object_name, object_path, confidence_score, bounding_box, created_at
            val2 = []
            for i in range(0, len(boxes)):
                width = boxes[i][2] - boxes[i][0]
                height = boxes[i][3] - boxes[i][1] 
                if(width < 25 and height < 25):
                    continue

                box = json.dumps({"x":boxes[i][0], "y":boxes[i][1], "width":width, "height":height})
                val2.append((maxObjId, maxImgId, "YOLOv3", names[i], path_to_watch1+"/"+f, confs[i], box,   created_at))
                obj = image[boxes[i][1]:boxes[i][3], boxes[i][0]:boxes[i][2]]
                cv2.imwrite(assetsDir+'/{img_id}-{obj_id}.jpg'.format(img_id = maxImgId, obj_id = maxObjId), obj)
                maxObjId += 1
            
            mycursor.executemany(sql4, val2)
            mydb.commit()

            safe_texts = generate_image_caption(image_path_or_url = path_to_watch1+"/"+f, model_id = "llava-hf/llava-1.5-7b-hf")
            
            val3 = []
            for t in safe_texts:
                val3.append((maxCapId, maxImgId, t, "llava-hf/llava-1.5-7b-hf", created_at))
                maxCapId += 1

            mycursor.executemany(sql6, val3)
            mydb.commit()    

            maxImgId += 1
            print('Done')

    # if removed1: print "Removed: ", ", ".join (removed1)
    
    added2 = [f for f in after2 if not f in before2]

    if added2: 
        print("Added: ", ", ".join(added2))
        for f in added2:
            image = cv2.imread(path_to_watch2+"/"+f)
            boxes,names, confs = predict_and_detect(model, image, classes=[], conf=0.25)
            # image_id, image_path, prompt_text, generation_time, user_id
            created_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            val1 = (maxImgId, path_to_watch2+"/"+f, "", created_at, 4)
            mycursor.execute(sql3, val1)
            mydb.commit()
            
            # object_id, image_id, model_used, object_name, object_path, confidence_score, bounding_box, created_at
            val2 = []
            for i in range(0, len(boxes)):
                width = boxes[i][2] - boxes[i][0]
                height = boxes[i][3] - boxes[i][1] 
                if(width < 25 and height < 25):
                    continue

                box = json.dumps({"x":boxes[i][0], "y":boxes[i][1], "width":width, "height":height})
                val2.append((maxObjId, maxImgId, "YOLOv3", names[i], path_to_watch2+"/"+f, confs[i], box,   created_at))
                obj = image[boxes[i][1]:boxes[i][3], boxes[i][0]:boxes[i][2]]
                cv2.imwrite(assetsDir+'/{img_id}-{obj_id}.jpg'.format(img_id = maxImgId, obj_id = maxObjId), obj)
                maxObjId += 1
            
            mycursor.executemany(sql4, val2)
            mydb.commit()

            safe_texts = generate_image_caption(image_path_or_url = path_to_watch2+"/"+f, model_id = "llava-hf/llava-1.5-7b-hf")

            val3 = []
            for t in safe_texts:
                val3.append((maxCapId, maxImgId, t, "llava-hf/llava-1.5-7b-hf", created_at))
                maxCapId += 1

            mycursor.executemany(sql6, val3)
            mydb.commit()     

            maxImgId += 1
            print('Done')

    added3 = [f for f in after3 if not f in before3]

    if added3: 
        print("Added: ", ", ".join(added3))
        for f in added3:
            image = cv2.imread(path_to_watch3+"/"+f)
            boxes,names, confs = predict_and_detect(model, image, classes=[], conf=0.25)
            # image_id, image_path, prompt_text, generation_time, user_id
            created_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            val1 = (maxImgId, path_to_watch3+"/"+f, "", created_at, 4)
            mycursor.execute(sql3, val1)
            mydb.commit()
            
            # object_id, image_id, model_used, object_name, object_path, confidence_score, bounding_box, created_at
            val2 = []
            for i in range(0, len(boxes)):
                width = boxes[i][2] - boxes[i][0]
                height = boxes[i][3] - boxes[i][1] 
                if(width < 25 and height < 25):
                    continue

                box = json.dumps({"x":boxes[i][0], "y":boxes[i][1], "width":width, "height":height})
                val2.append((maxObjId, maxImgId, "YOLOv3", names[i], path_to_watch3+"/"+f, confs[i], box, created_at))
                obj = image[boxes[i][1]:boxes[i][3], boxes[i][0]:boxes[i][2]]
                cv2.imwrite(assetsDir+'/{img_id}-{obj_id}.jpg'.format(img_id = maxImgId, obj_id = maxObjId), obj)
                maxObjId += 1
            
            mycursor.executemany(sql4, val2)
            mydb.commit()

            safe_texts = generate_image_caption(image_path_or_url = path_to_watch3+"/"+f, model_id = "llava-hf/llava-1.5-7b-hf")
            
            val3 = []
            for t in safe_texts:
                val3.append((maxCapId, maxImgId, t, "llava-hf/llava-1.5-7b-hf", created_at))
                maxCapId += 1

            mycursor.executemany(sql6, val3)
            mydb.commit() 

            maxImgId += 1
            print('Done') 

    before1 = after1
    before2 = after2
    before3 = after3                           

    now = datetime.now()
    
    formatted_date = now.strftime('%Y-%m-%d')
    
    if(path_to_watch1 != currentDir+"/output/txt2img-images/"+formatted_date and 
    dict ([(f, None) for f in os.listdir (path_to_watch1)]) == before1 and 
    before2 == dict ([(f, None) for f in os.listdir (path_to_watch2)]) ):

        isExist = os.path.exists(path_to_watch1)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(path_to_watch1)

        isExist = os.path.exists(path_to_watch2)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(path_to_watch2)  

        path_to_watch1 = currentDir+"/output/txt2img-images/"+formatted_date
        path_to_watch2 = currentDir+"/output/img2img-images/"+formatted_date

        before1 = {}
        before2 = {}