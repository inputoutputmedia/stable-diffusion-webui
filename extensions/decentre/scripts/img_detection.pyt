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

def generate_image_caption2(image_path_or_url: str, model_id: str) -> List[str]:
    try:
        question = "What is your assessment for this image?"
        prompt = f"A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.###Human: <image>\n{question}###Assistant:"
        raw_image = load_image(image_path_or_url)
        processor = AutoProcessor.from_pretrained(model_id)
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",bnb_4bit_compute_dtype=torch.float16)
        model = VipLlavaForConditionalGeneration.from_pretrained(model_id, low_cpu_mem_usage=True, quantization_config=quantization_config)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        inputs = processor(prompt, raw_image, return_tensors='pt').to(device, torch.float16)
        output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
        captions_text = processor.decode(output[0][2:], skip_special_tokens=True)
        captions_cleaned = captions_text[len(prompt):]
        sentences = [sentence.strip() for sentence in re.split(r'(?<=[.])\s+', captions_cleaned) if sentence]

        safe_texts = []
        for sentence in sentences:
            safe_text = sentence.replace("'", "''")
            safe_texts.append(safe_text)
        
        return safe_texts
    except Exception as e:
        logger.error(f"Error generating caption: {e}")
        raise

def generate_image_caption1(image_path_or_url: str, model_id: str) -> List[str]:
    try:
        raw_image = load_image(image_path_or_url)
        processor = LlavaNextProcessor.from_pretrained(model_id)
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",bnb_4bit_compute_dtype=torch.float16)
        model = LlavaNextForConditionalGeneration.from_pretrained(model_id, low_cpu_mem_usage=True, quantization_config=quantization_config)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        prompt = "[INST] <image>\nWhat is your assessment for this image? [/INST]"
        inputs = processor(prompt, raw_image, return_tensors='pt').to(device, torch.float16)
        
        output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
        captions_text = processor.decode(output[0], skip_special_tokens=True)
        
        captions_cleaned = captions_text[len(prompt)-5:]
        sentences = [sentence.strip() for sentence in re.split(r'(?<=[.])\s+', captions_cleaned) if sentence]

        safe_texts = []
        for sentence in sentences:
            safe_text = sentence.replace("'", "''")
            safe_texts.append(safe_text)
        
        return safe_texts
    except Exception as e:
        logger.error(f"Error generating caption: {e}")
        raise

def generate_image_caption0(image_path_or_url: str, model_id: str) -> List[str]:
    try:
        raw_image = load_image(image_path_or_url)
        # specify how to quantize the model
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",bnb_4bit_compute_dtype=torch.float16)
        model = LlavaForConditionalGeneration.from_pretrained(model_id, low_cpu_mem_usage=True, quantization_config=quantization_config)
        processor = AutoProcessor.from_pretrained(model_id)

        prompt = "USER: <image>\nWhat is your assessment for this image?\nASSISTANT:"
        inputs = processor(prompt, raw_image, return_tensors='pt').to("cuda" if torch.cuda.is_available() else "cpu", torch.float16)

        output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
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

def generate_image_caption(image_path_or_url: str, model_id: str) -> List[str]:
        if model_id == "llava-v1.6-mistral-7b-hf":
             return generate_image_caption1(image_path_or_url = image_path_or_url, model_id = "llava-hf/llava-v1.6-mistral-7b-hf")
        elif model_id == "llava-1.5-7b-hf":
            return generate_image_caption0(image_path_or_url = image_path_or_url, model_id = "llava-hf/llava-1.5-7b-hf")
        elif model_id == "vip-llava-7b-hf":
            return generate_image_caption2(image_path_or_url = image_path_or_url, model_id = "llava-hf/vip-llava-7b-hf")
        
        return []

currentDir = str(Path(os.getcwd()).absolute())

# with open(currentDir+"/mysql_connection_input.txt", 'r') as file:
#     lines = [line.rstrip() for line in file]

# mydb = mysql.connector.connect(
#     host=lines[2].split('=')[1],
#     user=lines[3].split('=')[1],
#     password=lines[4].split('=')[1],
#     database=lines[5].split('=')[1])

assetsDir = os.getenv('LOCALAPPDATA')+"/Decentre/assets"

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

dbFile = os.getenv('LOCALAPPDATA')+"/Decentre/decentre.db"
db = sqlite3.connect(dbFile)
mycursor = db.cursor()

def detect_image(image_path_or_url: str, isCaption: str, model_path: str):
    sql1 = "SELECT IFNULL(MAX(image_id), 0) FROM image_table"
    sql2 = "SELECT IFNULL(MAX(object_id), 0) FROM object_detection_table"
    sql3 = "INSERT INTO image_table (image_id, image_path, prompt_text, generation_time, user_id) VALUES (?, ?, ?, ?, ?)"
    sql4 = "INSERT INTO object_detection_table (object_id, image_id, model_used, object_name, object_path, confidence_score, bounding_box, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
    sql5 = "SELECT IFNULL(MAX(caption_id), 0) FROM captions_table"
    sql6 = "INSERT INTO captions_table (caption_id, image_id, text, model_used, created_at) VALUES (?, ?, ?, ?, ?)"
    sql7 = "SELECT IFNULL(MAX(object_caption_id), 0) FROM object_captions"
    sql8 = "INSERT INTO object_captions (object_caption_id, object_id, image_id, caption_text, model_used, created_at) VALUES (?, ?, ?, ?, ?, ?)"

    mycursor.execute(sql1)
    maxImgId = mycursor.fetchone()[0] + 1
    mycursor.execute(sql2)
    maxObjId = mycursor.fetchone()[0] + 1
    mycursor.execute(sql5)
    maxCapId = mycursor.fetchone()[0] + 1
    mycursor.execute(sql7)
    maxObjCapId = mycursor.fetchone()[0] + 1
  
    if isCaption == "True":
        safe_texts = generate_image_caption(image_path_or_url=image_path_or_url, model_id=model_path)

        created_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
        val3 = []
        #print("cap detection:{}".format(len(safe_texts)))
        for t in safe_texts:
            if len(t) < capLen:
                continue

            val3.append((maxCapId, maxImgId, t, model_path, created_at))
            maxCapId += 1

        mycursor.executemany(sql6, val3)
        db.commit()

        print("caps detected")

        val4 = []
        sql9 = "SELECT object_id, bounding_box fROM object_detection_table WHERE image_id = {}".format(maxImgId)
        mycursor.execute(sql9)
        objs = mycursor.fetchall()
        for i in range(0, len(objs)):

            safe_texts = generate_image_caption(image_path_or_url = assetsDir+'/{img_id}-{obj_id}.jpg'.format(img_id = maxImgId, obj_id = objs[i][0]), model_id = model_path)
            safe_txt_List = []
            for t in safe_texts:
                if len(t) >= capLen:
                    safe_txt_List.append(t)

            created_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            val4.append((maxObjCapId, objs[i][0], maxImgId, json.dumps(safe_txt_List), model_path, created_at))
        
            maxObjCapId += 1

        mycursor.executemany(sql8, val4)
        db.commit()

        print("obj caps detected")

    else:
    # if image_path_or_url.startswith(('http://', 'https://')):
    #     resp = urlopen(image_path_or_url)
    #     image = np.asarray(bytearray(resp.read()), dtype="uint8")
    #     image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    # else:
        model = YOLO(model_path)
        image = cv2.imread(image_path_or_url)
        boxes,names, confs = predict_and_detect(model, image, classes=[], conf=confl)
        # image_id, image_path, prompt_text, generation_time, user_id
        created_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        val1 = (maxImgId, image_path_or_url[len(os.getenv('LOCALAPPDATA')):], "", created_at, 4)
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
            val2.append((maxObjId, maxImgId, model_path, names[i], assetsDir[len(os.getenv('LOCALAPPDATA')):]+'/{img_id}-{obj_id}.jpg'.format(img_id = maxImgId, obj_id = maxObjId), confs[i], box,   created_at))
            obj = image[boxes[i][1]:boxes[i][3], boxes[i][0]:boxes[i][2]]
            cv2.imwrite(assetsDir+'/{img_id}-{obj_id}.jpg'.format(img_id = maxImgId, obj_id = maxObjId), obj)
            maxObjId += 1
    
        mycursor.executemany(sql4, val2)
        db.commit()

        print("objs detected")
    db.close()    

detect_image(sys.argv[1], sys.argv[2], sys.argv[3]) 