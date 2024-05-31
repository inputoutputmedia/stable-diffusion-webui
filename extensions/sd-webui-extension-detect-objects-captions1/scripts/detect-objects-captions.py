import modules.scripts as script
import gradio as gr
from modules import images, script_callbacks
from modules.processing import process_images, Processed
from modules.processing import Processed
from modules.shared import opts, cmd_opts, state
from PIL import Image
from ultralytics import YOLO
import time 
import requests
from io import BytesIO
import os
import subprocess
# from selenium import webdriver
# from selenium.webdriver import ChromeOptions
from seleniumbase import SB
from bs4 import BeautifulSoup
import urllib.request
from pathlib import Path
from datetime import datetime
import time
import shutil
import sqlite3

# class DetectionScript(script.Script):
#         def __init__(self) -> None:
#                 super().__init__()

#         # Extension title in menu UI
#         def title(self):
#                 return "Objects Detections and Captions"
        
#         def show(self, is_img2img):
#                 return script.AlwaysVisible
        
#         def ui(self, is_img2img):
#                 return ()

#         def load_settings(_,self):
#              settingsFile = "extensions/sd-webui-extension-detect-objects-captions1/settings.st"
#              global objWidth   
#              global objHeight 
#              global confl 
#              global capLen 

#              file = open(settingsFile, 'r')

#              objWidth = int(file.readline()[18:])
#              objHeight = int(file.readline()[19:])
#              confl = float(file.readline()[22:])
#              capLen = int(file.readline()[20:])

#              file.close()
           

#         script_callbacks.on_app_started(load_settings)

detectProcessDurations = None

def multi_img_detection(dm = "yolov3u.pt", progress=gr.Progress()):
    global addedImgIndex
    global added_imgs
    global imgclicked
    global detectProcessDurations
    global totalDetDur
    if len(added_imgs) > 0:
        sql1 = "SELECT IFNULL(MAX(image_id), 0) FROM image_table"
        mycursor.execute(sql1)
        maxImgId = mycursor.fetchone()[0] + 1
        start = time.time()
        detectProcessDurations = []
        endTimes = [start]
        #processes = ["Object Detection"]
        for i in progress.tqdm(range(0,len(added_imgs)), desc="Processing..."):
                op = subprocess.Popen(["python", "extensions/sd-webui-extension-detect-objects-captions1/scripts/multi_img_detection.pyt", added_imgs[i], "False", dm, str(maxImgId)], stdout=subprocess.PIPE, universal_newlines=True)
                iter_line = iter(op.stdout.readline, "") 
                while(True):
                    stdout_line = next(iter_line)
                    if stdout_line.startswith("objs detected"):
                       detectProcessDurations.append(time.time() - endTimes[-1])
                       endTimes.append(time.time())   
                       break
                maxImgId += 1        

        totalDetDur = endTimes[-1] - start
        # with open(os.getenv('LOCALAPPDATA')+"/sd-webui-extension/Images_detected_log.txt", "a") as myfile:
        #     myfile.write("Number of Images: {} \n".format(len(added_imgs)))
        #     for i in range(0,len(added_imgs)):
        #          myfile.write("Image {index} location: {url} \n".format(index = i+1, url = added_imgs[i])) 
        #          for j in range (0,3):
        #              myfile.write("{process} Duration: {dur}s\n".format(process = processes[j], dur = processDurations[i*3+j]))
        #     totalDur = endTimes[-1] - start
        #     myfile.write("Total Duration: {dur}s \n".format(dur = totalDur))
        #     myfile.close()         
                
        # addedImgIndex = 0
        # imgclicked = -1
        # added_imgs = []              
        # processing = False        
        #return [None for i in range(0, 48)] + [None, "Detection Done", "Captions Done"]
        return "Detection Done"
    #return [None for i in range(0, 48)] + [None, "Error", "Add Images first"]
    return "Error"

def multi_img_caption(cm = "llava-1.5-7b-hf", progress=gr.Progress()):
    global addedImgIndex
    global added_imgs
    global imgclicked
    global totalDetDur
    if len(added_imgs) > 0:
        sql1 = "SELECT IFNULL(MAX(image_id), 0) FROM image_table"
        mycursor.execute(sql1)
        maxImgId = mycursor.fetchone()[0] + 1        
        start = time.time()
        processDurations = []
        endTimes = [start]
        processes = ["Object Detection", "Caption Detection", "Object Caption Detection"]
        for i in progress.tqdm(range(0,len(added_imgs)), desc="Processing..."):
                op = subprocess.Popen(["python", "extensions/sd-webui-extension-detect-objects-captions1/scripts/multi_img_detection.pyt", added_imgs[i], "True", cm, str(maxImgId)], stdout=subprocess.PIPE, universal_newlines=True)
                iter_line = iter(op.stdout.readline, "") 
                for j in range(0,2):   
                    while(True):
                        stdout_line = next(iter_line)
                        if stdout_line.startswith("caps detected"):
                            processDurations.append(time.time() - endTimes[-1])
                            endTimes.append(time.time()) 
                            break
                        if stdout_line.startswith("obj caps detected"):
                            processDurations.append(time.time() - endTimes[-1])
                            endTimes.append(time.time()) 
                            break
                maxImgId += 1         

        with open(os.getenv('LOCALAPPDATA')+"/sd-webui-extension/Images_detected_log.txt", "a") as myfile:
            myfile.write("Number of Images: {} \n".format(len(added_imgs)))
            for i in range(0,len(added_imgs)):
                 myfile.write("Image {index} location: {url} \n".format(index = i+1, url = added_imgs[i])) 
                 myfile.write("{process} Duration: {dur}s\n".format(process = processes[0], dur = detectProcessDurations[i]))
                 for j in range (0,2):
                     myfile.write("{process} Duration: {dur}s\n".format(process = processes[j+1], dur = processDurations[i*2+j]))
            totalDur = totalDetDur + endTimes[-1] - start
            totalDetDur = 0
            myfile.write("Total Duration: {dur}s \n".format(dur = totalDur))
            myfile.close()         
                
        addedImgIndex = 0
        imgclicked = -1
        added_imgs = []              
     
        #return [None for i in range(0, 48)] + [None, "Detection Done", "Captions Done"]
        return "Captions Done"
    #return [None for i in range(0, 48)] + [None, "Error", "Add Images first"]
    return "Add Images first"

def reset_images():
    global added_imgs
    global addedImgIndex
    global imgclicked
    added_imgs = []
    addedImgIndex = 0
    imgclicked = -1
    if len(imgs) > 0:
        return [load_image(imgs[imgIndex])] + [None for i in range(0, 24)]
    return [None] + [None for i in range(0, 24)]

def cap_detection(cm = "llava-1.5-7b-hf", progress=gr.Progress()):
    global totalDetDur
    if imgclicked != -1:
        start = time.time()
        processDurations = []
        endTimes = [start]
        processes = ["Caption Detection", "Object Caption Detection"]
        op = subprocess.Popen(["python", "extensions/sd-webui-extension-detect-objects-captions1/scripts/img_detection.pyt", added_imgs[imgclicked], "True", cm], stdout=subprocess.PIPE, universal_newlines=True)
        iter_line = iter(op.stdout.readline, "")     
        for i in progress.tqdm(range(0,2), desc="Processing..."):
            while(True):
                stdout_line = next(iter_line)
                if stdout_line.startswith("caps detected"):
                     processDurations.append(time.time() - endTimes[-1])
                     endTimes.append(time.time()) 
                     break
                if stdout_line.startswith("obj caps detected"):
                     processDurations.append(time.time() - endTimes[-1])
                     endTimes.append(time.time()) 
                     break
        with open(os.getenv('LOCALAPPDATA')+"/sd-webui-extension/Images_detected_log.txt", "a") as myfile:
            #myfile.write("Number of Images: {} \n".format(1))
            #myfile.write("Image {index} location: {url} \n".format(index = 1, url = added_imgs[imgclicked])) 
            for j in range (0,2):
               myfile.write("{process} Duration: {dur}s\n".format(process = processes[j], dur = processDurations[j]))
            totalDur = totalDetDur + endTimes[-1] - start
            totalDetDur = 0
            myfile.write("Total Duration: {dur}s \n".format(dur = totalDur))
            myfile.close()         
                     
        return "Captions Done"
    
    return "Select Image"

def img_detection(dm = "yolov3u.pt", progress=gr.Progress()):
    global totalDetDur
    if imgclicked != -1:
        start = time.time()
        processDurations = []
        endTimes = [start]
        processes = ["Object Detection"]
        op = subprocess.Popen(["python", "extensions/sd-webui-extension-detect-objects-captions1/scripts/img_detection.pyt", added_imgs[imgclicked], "False", dm], stdout=subprocess.PIPE, universal_newlines=True)
        iter_line = iter(op.stdout.readline, "")     
        for i in progress.tqdm(range(0,1), desc="Processing..."):
            while(True):
                stdout_line = next(iter_line)
                if stdout_line.startswith("objs detected"):
                     processDurations.append(time.time() - endTimes[-1])
                     endTimes.append(time.time()) 
                     break

        with open(os.getenv('LOCALAPPDATA')+"/sd-webui-extension/Images_detected_log.txt", "a") as myfile:
            myfile.write("Number of Images: {} \n".format(1))
            myfile.write("Image {index} location: {url} \n".format(index = 1, url = added_imgs[imgclicked])) 
            for j in range (0,1):
               myfile.write("{process} Duration: {dur}s\n".format(process = processes[j], dur = processDurations[j]))
            totalDetDur = endTimes[-1] - start
            # myfile.write("Total Duration: {dur}s \n".format(dur = totalDur))
            myfile.close()         
                      
        return "Detection Done"
    
    return "Error"

def update_images(image_directory = "", progress=gr.Progress()):

        if os.path.exists(image_directory):       
            image_files = [f for f in os.listdir(image_directory) if f.endswith((".png", ".jpg", ".jpeg"))]
            global imgs
            global imgIndex
            global addedImgIndex
            global added_imgs
            global imgclicked
            imgIndex = 0
            addedImgIndex = 0
            imgclicked = -1
            added_imgs = []
            imgs = []
            importedImagesFolder = os.getenv('LOCALAPPDATA')+"/sd-webui-extension/imported_images"
            importedImagesFolder = importedImagesFolder + "/" + datetime.now().strftime('%Y-%m-%d %H-%M-%S')    
            os.makedirs(importedImagesFolder) 
            for image_file in progress.tqdm(image_files, desc="Processing..."):
                shutil.copyfile(os.path.join(image_directory, image_file), os.path.join(importedImagesFolder, image_file))
                imgs.append(os.path.join(importedImagesFolder, image_file))
   
            if len(imgs) > 0:
                if len(imgs) < 24:
                    return [load_image(imgs[0])] + [load_image(imgs[i]) for i in range(0, len(imgs))] + [None for i in range(len(imgs), 48)] + ["Import Done"]
                else:        
                    return [load_image(imgs[0])] + [load_image(imgs[i]) for i in range(0, 24)] + [None for i in range(24, 48)] + ["Import Done"]
        return [None] + [None for i in range(0, 48)] + ["Import Done"]

def load_image(image_path_or_url: str) -> Image.Image:
    image = None
    if image_path_or_url.startswith(('http://', 'https://')):
        response = requests.get(image_path_or_url)
        image = Image.open(BytesIO(response.content))
    elif image_path_or_url != "":
        image = Image.open(image_path_or_url)
    return image 

imgs = []
imgIndex = 0
addedImgIndex = 0
added_imgs = []
imgclicked = -1
totalDetDur = 0
currentDir = str(Path(os.getcwd()).absolute())

def minus():
    global imgIndex
    global imgclicked
    imgclicked = -1
    if imgIndex > 0:
        imgIndex = imgIndex - 1
    if len(imgs) > 0 :    
        return load_image(imgs[imgIndex])
    else:
         return None       

def plus():
     global imgIndex
     global imgclicked
     imgclicked = -1
     if imgIndex < len(imgs)-1 and imgIndex <23:
          imgIndex = imgIndex + 1
     if len(imgs) > 0:    
        return load_image(imgs[imgIndex])
     else:
         return None      

def add_image():
    global addedImgIndex
    global imgclicked
    global imgIndex
    global imgs
    if len(imgs) > 0 and addedImgIndex < 24:
        added_imgs.append(imgs[imgIndex])
        imgclicked = addedImgIndex
        addedImgIndex += 1
        imgs = imgs[0:imgIndex]+imgs[imgIndex+1:len(imgs)]
        if imgIndex > 0:
            imgIndex = imgIndex - 1
        if len(imgs) < 24:
           if len(imgs) > 0:
               return [load_image(imgs[imgIndex])] + [load_image(imgs[i]) for i in range(0, len(imgs))] + [None for i in range(len(imgs), 24)] + [load_image(added_imgs[i]) for i in range(0, len(added_imgs))] + [None for i in range(len(added_imgs), 24)]
           return [None] + [None for i in range(0, 24)] + [load_image(added_imgs[i]) for i in range(0, len(added_imgs))] + [None for i in range(len(added_imgs), 24)]
        else:
           return [load_image(imgs[imgIndex])] + [load_image(imgs[i]) for i in range(0, 24)] + [load_image(added_imgs[i]) for i in range(0, len(added_imgs))] + [None for i in range(len(added_imgs), 24)]
    elif len(imgs) > 0:
        if len(imgs) < 24:
            return [load_image(imgs[imgIndex])] + [load_image(imgs[i]) for i in range(0, len(imgs))] + [None for i in range(len(imgs), 24)] + [load_image(added_imgs[i]) for i in range(0, 24)]
        else:
           return [load_image(imgs[imgIndex])] + [load_image(imgs[i]) for i in range(0, 24)] + [load_image(added_imgs[i]) for i in range(0, 24)]
    else:
        return [None] + [None for i in range(0, 24)] + [load_image(added_imgs[i]) for i in range(0, len(added_imgs))] + [None for i in range(len(added_imgs), 24)]

def remove_image():
   global addedImgIndex
   global added_imgs
   global imgclicked
   global imgIndex
   global imgs
   if(len(imgs) > 0 and addedImgIndex > 0 and imgclicked != -1):
       added_imgs = added_imgs[0:imgclicked]+added_imgs[imgclicked+1:addedImgIndex]
       addedImgIndex -= 1
       imgclicked = -1
       if len(imgs) < 24:
           if len(imgs) > 0:
               return [load_image(imgs[imgIndex])] + [load_image(imgs[i]) for i in range(0, len(imgs))] + [None for i in range(len(imgs), 24)] + [load_image(added_imgs[i]) for i in range(0, len(added_imgs))] + [None for i in range(len(added_imgs), 24)]
           return [None] + [None for i in range(0, 24)] + [load_image(added_imgs[i]) for i in range(0, len(added_imgs))] + [None for i in range(len(added_imgs), 24)]
       else:
          return [load_image(imgs[imgIndex])] + [load_image(imgs[i]) for i in range(0, 24)] + [load_image(added_imgs[i]) for i in range(0, len(added_imgs))] + [None for i in range(len(added_imgs), 24)]    
   elif len(imgs) > 0:
      imgs = imgs[0:imgIndex]+imgs[imgIndex+1:len(imgs)]
      if imgIndex > 0:
            imgIndex = imgIndex - 1
      if len(imgs) < 24:
            if len(imgs) > 0:
                return [load_image(imgs[imgIndex])] + [load_image(imgs[i]) for i in range(0, len(imgs))] + [None for i in range(len(imgs), 24)] + [load_image(added_imgs[i]) for i in range(0, len(added_imgs))] + [None for i in range(len(added_imgs), 24)]
            return [None] + [None for i in range(0, 24)] + [load_image(added_imgs[i]) for i in range(0, len(added_imgs))] + [None for i in range(len(added_imgs), 24)]
      else:
           return [load_image(imgs[imgIndex])] + [load_image(imgs[i]) for i in range(0, 24)] + [load_image(added_imgs[i]) for i in range(0, len(added_imgs))] + [None for i in range(len(added_imgs), 24)]      
   else:
       return [None] + [None for i in range(0, 24)] + [load_image(added_imgs[i]) for i in range(0, len(added_imgs))] + [None for i in range(len(added_imgs), 24)]    
       
def img_clicked(img_index):
   global imgclicked
   global imgIndex
   if(img_index < 24):
      imgIndex = img_index
      imgclicked = -1
      return load_image(imgs[img_index])
   else:   
      imgclicked = img_index-24
      return load_image(added_imgs[img_index-24])

def reset_objsettings():
     return [25,25,0.25]

def reset_capsettings():
     return 25

def save_settings(obj_width, obj_height,_conf_level, cap_length, progress=gr.Progress()):
     #Min object width: 25
     #Min object height: 25
     #Min confidence level: 0.25
     #Min caption length: 25
     settings = ["Min object width: {} \n".format(obj_width), "Min object height: {} \n".format(obj_height), 
                 "Min confidence level: {} \n".format(_conf_level), "Min caption length: {} \n".format(cap_length)]

     file = open('extensions/sd-webui-extension-detect-objects-captions1/settings.st', 'w')

     # Writing settings
     for i in progress.tqdm(range(4), desc="Processing..."):
             file.write(settings[i])
             time.sleep(0.1)
 
     # Closing file
     file.close()
         
     return "settings saved"
                      
def label_rest():
     for i in range(0, 3):
          time.sleep(1)
     return ""

def no_fn():
    return

def import_from_url(url, progress=gr.Progress()):
    if (not url.startswith(('http://', 'https://'))):
         return

#     options = ChromeOptions()
#     options.add_argument("--headless=new")
#     driver = webdriver.Chrome(options=options)
#     driver.get(url)
#     content = driver.page_source
#     soup = BeautifulSoup(content, "html.parser")
#     driver.quit()
    global imgs
    global imgIndex
    global addedImgIndex
    global added_imgs
    global imgclicked
    imgIndex = 0
    addedImgIndex = 0
    imgclicked = -1
    added_imgs = []
    imgs = []
    #imgall=soup.find_all('img')


    downloadDir = os.getenv('LOCALAPPDATA')+"/sd-webui-extension/imported_images"

    downloadDir = downloadDir + "/" + datetime.now().strftime('%Y-%m-%d %H-%M-%S')    
    os.makedirs(downloadDir)    

        # i = len(os.listdir(downloadDir))+1

#     headers={'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
#    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
#    'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
#    'Accept-Encoding': 'none',
#    'Accept-Language': 'en-US,en;q=0.8',
#    'Connection': 'keep-alive'}    

#     for img in progress.tqdm(imgall, desc="Processing..."):
#         try:
#             imgsrc=img['data-srcset']
#         except:
#             try:
#                 imgsrc=img['data-src']
#             except:
#                 try:
#                     imgsrc=img['data-fallback-src']
#                 except:
#                     try:
#                         imgsrc=img['src']
#                     except:
#                         pass
#         if not 'svg' in imgsrc:
#             if 'jpg' in imgsrc or 'jpeg' in imgsrc:
#                 if imgsrc.startswith("//www."):
#                     imgsrc = "https:"+imgsrc
#                 elif imgsrc.startswith("/"):                
#                     imgsrc = url+imgsrc
#                 request_=urllib.request.Request(imgsrc,None,headers) #The assembled request
#                 response = urllib.request.urlopen(request_)# store the response
#                 f = open(downloadDir+"/image-{}.jpg".format(i),'wb')
#                 f.write(response.read())
#                 f.close()
#                 imgs.append(downloadDir+"/image-{}.jpg".format(i))
#                 i=i+1

#             elif 'png' in imgsrc:
#                 if imgsrc.startswith("//www."):
#                     imgsrc = "https:"+imgsrc
#                 elif imgsrc.startswith("/"):                
#                     imgsrc = url+imgsrc
#                 request_=urllib.request.Request(imgsrc,None,headers) #The assembled request
#                 response = urllib.request.urlopen(request_)# store the response
#                 f = open(downloadDir+"/image-{}.png".format(i),'wb')
#                 f.write(response.read())
#                 f.close()
#                 imgs.append(downloadDir+"/image-{}.png".format(i))
#                 i=i+1 
    with SB() as sb:
        sb.open(url)
        img_elements_with_src = sb.find_elements("img[src]")
        unique_src_values = []
        for img in img_elements_with_src:
            src = img.get_attribute("src")
            if src not in unique_src_values:
                unique_src_values.append(src)

        for src in progress.tqdm(unique_src_values, desc="Processing..."):
            ext = [ele for ele in ["png", "jpeg", "jpg"] if (ele in src)]
            if(len(ext) == 0):
                continue
            
            num = 3
            if("jpeg" in ext[0]):
                num = 4
            
            start = "http:"
            if("https:" in src):
               start = "https:"

            src1 =  src[src.find(start):(src.find(ext[0])+num)]  
            
            sb.download_file(src1, downloadDir)
            filename = src1.split("/")[-1]
            #sb.assert_downloaded_file(filename)
            file_path = os.path.join(downloadDir, filename)
            imgs.append(file_path)
            time.sleep(0.1)      
   
    if len(imgs) > 0:
        if len(imgs) < 24:
            return [load_image(imgs[0])] + [load_image(imgs[i]) for i in range(0, len(imgs))] + [None for i in range(len(imgs), 48)] + ["Import Done"]
        else:        
            return [load_image(imgs[0])] + [load_image(imgs[i]) for i in range(0, 24)] + [None for i in range(24, 48)] + ["Import Done"]
    return [None] + [None for i in range(0, 48)] + ["Import Done"]             

def isSQLite3(filename):
    from os.path import isfile, getsize

    if not isfile(filename):
        return False
    if getsize(filename) < 100: # SQLite database file header is 100 bytes
        return False

    with open(filename, 'rb') as fd:
        header = fd.read(100)

    return header[0:16] == b'SQLite format 3\000'

mycursor = None
     
def on_ui_tabs():
  subprocess.call(['pip', 'install', '-r', 'requirements_for_image_detection_and_caption.txt'])
  YOLO("yolov3u.pt")
  YOLO("yolov5x6u.pt")
  YOLO("yolov8x.pt")
  global mycursor
  label1 = None
  label2 = None
  added_imgs = [None] * 48
  settingsFile = "extensions/sd-webui-extension-detect-objects-captions1/settings.st"

  file = open(settingsFile, 'r')

  objWidth = int(file.readline()[18:])
  objHeight = int(file.readline()[19:])
  confl = float(file.readline()[22:])
  capLen = int(file.readline()[20:])
  
  importedImagesFolder = os.getenv('LOCALAPPDATA')+"/sd-webui-extension/imported_images"
  isExist = os.path.exists(importedImagesFolder)
  if not isExist:
    # Create a new directory because it does not exist
    os.makedirs(importedImagesFolder)


  sqlFile = os.getenv('LOCALAPPDATA')+"/sd-webui-extension/sdwebui_extention_local.sql"
  dbFile = os.getenv('LOCALAPPDATA')+"/sd-webui-extension/sdwebui_extention_local.db"
  sql_script = ""  

  if(not isSQLite3(dbFile)):
      with open(sqlFile, 'r') as sql_file:
          sql_script = sql_file.read()
          sql_file.close()
      db = sqlite3.connect(dbFile, check_same_thread=False)
      mycursor = db.cursor()
      mycursor.executescript(sql_script)
      db.commit()
  else:
      db = sqlite3.connect(dbFile, check_same_thread=False)
      mycursor = db.cursor()

  with gr.Blocks(analytics_enabled=False) as Objects_Detections_and_Captions:
    with gr.Column(elem_id = "columnMain"):
      gr.Image("extensions/sd-webui-extension-detect-objects-captions1/decenter_studio.jpeg", height = 57, width = 160, show_label = False, container = False, show_download_button = False)
      with gr.Row():
        with gr.Column(elem_classes = "panel3"):
          with gr.Row(elem_classes = "panel1", min_width=550):
            image = gr.Image(height = 1000, width = 550, show_label = False, container = True, show_download_button = False)
          with gr.Row(min_width=550): 
            less = gr.Button(value="<", variant="primary", size="sm", min_width = 75, scale = 1, elem_id = "buttonLeft") 
            addImg = gr.Button(value="Add", variant="primary", size="sm", min_width = 150,  scale = 2, elem_id = "buttonAdd")
            removeImg = gr.Button(value="Remove", variant="primary",size="sm", min_width = 150, scale = 2, elem_id = "buttonRemove")
            greater = gr.Button(value=">", variant="primary", size="sm",  min_width = 75, scale = 1, elem_id = "buttonRight") 
          
          with gr.Row(elem_classes = "panel1", min_width=550):
              with gr.Row(elem_classes = "panelm", elem_id = "row0"):
                added_imgs[0] = gr.Image(height = 75, width = 75, min_width = 75, show_label = False, container = True, show_download_button = False, interactive = False)
              with gr.Row(elem_classes = "panelm", elem_id = "row1"):
                added_imgs[1] = gr.Image(height = 75, width = 75, min_width = 75, show_label = False, container = True, show_download_button = False, interactive = False)
              with gr.Row(elem_classes = "panelm", elem_id = "row2"):
                added_imgs[2] = gr.Image(height = 75, width = 75, min_width = 75, show_label = False, container = True, show_download_button = False, interactive = False)
              with gr.Row(elem_classes = "panelm", elem_id = "row3"):
                added_imgs[3] = gr.Image(height = 75, width = 75, min_width = 75, show_label = False, container = True, show_download_button = False, interactive = False)
              with gr.Row(elem_classes = "panelm", elem_id = "row4"):
                added_imgs[4] = gr.Image(height = 75, width = 75, min_width = 75, show_label = False, container = True, show_download_button = False, interactive = False)
              with gr.Row(elem_classes = "panelm", elem_id = "row5"):
                added_imgs[5] = gr.Image(height = 75, width = 75, min_width = 75, show_label = False, container = True, show_download_button = False, interactive = False)
          with gr.Row(elem_classes = "panel1", min_width=550):
              with gr.Row(elem_classes = "panelm", elem_id = "row6"):
                added_imgs[6] = gr.Image(height = 75, width = 75, min_width = 75, show_label = False, container = True, show_download_button = False, interactive = False)
              with gr.Row(elem_classes = "panelm", elem_id = "row7"):
                added_imgs[7] = gr.Image(height = 75, width = 75, min_width = 75, show_label = False, container = True, show_download_button = False, interactive = False)
              with gr.Row(elem_classes = "panelm", elem_id = "row8"):
                added_imgs[8] = gr.Image(height = 75, width = 75, min_width = 75, show_label = False, container = True, show_download_button = False, interactive = False)
              with gr.Row(elem_classes = "panelm", elem_id = "row9"):
                added_imgs[9] = gr.Image(height = 75, width = 75, min_width = 75, show_label = False, container = True, show_download_button = False, interactive = False)
              with gr.Row(elem_classes = "panelm", elem_id = "row10"):
                added_imgs[10] = gr.Image(height = 75, width = 75, min_width = 75, show_label = False, container = True, show_download_button = False, interactive = False)
              with gr.Row(elem_classes = "panelm", elem_id = "row11"):
                added_imgs[11] = gr.Image(height = 75, width = 75, min_width = 75, show_label = False, container = True, show_download_button = False, interactive = False)
          with gr.Row(elem_classes = "panel1", min_width=550):
              with gr.Row(elem_classes = "panelm", elem_id = "row12"):
                added_imgs[12] = gr.Image(height = 75, width = 75, min_width = 75, show_label = False, container = True, show_download_button = False, interactive = False)
              with gr.Row(elem_classes = "panelm", elem_id = "row13"):
                added_imgs[13] = gr.Image(height = 75, width = 75, min_width = 75, show_label = False, container = True, show_download_button = False, interactive = False)
              with gr.Row(elem_classes = "panelm", elem_id = "row14"):
                added_imgs[14] = gr.Image(height = 75, width = 75, min_width = 75, show_label = False, container = True, show_download_button = False, interactive = False)
              with gr.Row(elem_classes = "panelm", elem_id = "row15"):
                added_imgs[15] = gr.Image(height = 75, width = 75, min_width = 75, show_label = False, container = True, show_download_button = False, interactive = False)
              with gr.Row(elem_classes = "panelm", elem_id = "row16"):
                added_imgs[16] = gr.Image(height = 75, width = 75, min_width = 75, show_label = False, container = True, show_download_button = False, interactive = False)
              with gr.Row(elem_classes = "panelm", elem_id = "row17"):
                added_imgs[17] = gr.Image(height = 75, width = 75, min_width = 75, show_label = False, container = True, show_download_button = False, interactive = False)
          with gr.Row(elem_classes = "panel1", min_width=550):
              with gr.Row(elem_classes = "panelm", elem_id = "row18"):
                added_imgs[18] = gr.Image(height = 75, width = 75, min_width = 75, show_label = False, container = True, show_download_button = False, interactive = False)
              with gr.Row(elem_classes = "panelm", elem_id = "row19"):
                added_imgs[19] = gr.Image(height = 75, width = 75, min_width = 75, show_label = False, container = True, show_download_button = False, interactive = False)
              with gr.Row(elem_classes = "panelm", elem_id = "row20"):
                added_imgs[20] = gr.Image(height = 75, width = 75, min_width = 75, show_label = False, container = True, show_download_button = False, interactive = False)
              with gr.Row(elem_classes = "panelm", elem_id = "row21"):
                added_imgs[21] = gr.Image(height = 75, width = 75, min_width = 75, show_label = False, container = True, show_download_button = False, interactive = False)
              with gr.Row(elem_classes = "panelm", elem_id = "row22"):
                added_imgs[22] = gr.Image(height = 75, width = 75, min_width = 75, show_label = False, container = True, show_download_button = False, interactive = False)
              with gr.Row(elem_classes = "panelm", elem_id = "row23"):
                added_imgs[23] = gr.Image(height = 75, width = 75, min_width = 75, show_label = False, container = True, show_download_button = False, interactive = False)  
        with gr.Column(elem_classes = "panel3"):
          with gr.Column(elem_classes = "panel1", scale = 5):
            with gr.Row(min_width=550):
              with gr.Row(elem_classes = "panelm", elem_id = "row24"):
                added_imgs[24] = gr.Image(height = 75, width = 75, min_width = 75, show_label = False, container = True, show_download_button = False, interactive = False)
              with gr.Row(elem_classes = "panelm", elem_id = "row25"):
                added_imgs[25] = gr.Image(height = 75, width = 75, min_width = 75, show_label = False, container = True, show_download_button = False, interactive = False)
              with gr.Row(elem_classes = "panelm", elem_id = "row26"):
                added_imgs[26] = gr.Image(height = 75, width = 75, min_width = 75, show_label = False, container = True, show_download_button = False, interactive = False)
              with gr.Row(elem_classes = "panelm", elem_id = "row27"):
                added_imgs[27] = gr.Image(height = 75, width = 75, min_width = 75, show_label = False, container = True, show_download_button = False, interactive = False)
              with gr.Row(elem_classes = "panelm", elem_id = "row28"):
                added_imgs[28] = gr.Image(height = 75, width = 75, min_width = 75, show_label = False, container = True, show_download_button = False, interactive = False)
              with gr.Row(elem_classes = "panelm", elem_id = "row29"):
                added_imgs[29] = gr.Image(height = 75, width = 75, min_width = 75, show_label = False, container = True, show_download_button = False, interactive = False)
            with gr.Row(min_width=550):
              with gr.Row(elem_classes = "panelm", elem_id = "row30"):
                added_imgs[30] = gr.Image(height = 75, width = 75, min_width = 75, show_label = False, container = True, show_download_button = False, interactive = False)
              with gr.Row(elem_classes = "panelm", elem_id = "row31"):
                added_imgs[31] = gr.Image(height = 75, width = 75, min_width = 75, show_label = False, container = True, show_download_button = False, interactive = False)
              with gr.Row(elem_classes = "panelm", elem_id = "row32"):
                added_imgs[32] = gr.Image(height = 75, width = 75, min_width = 75, show_label = False, container = True, show_download_button = False, interactive = False)
              with gr.Row(elem_classes = "panelm", elem_id = "row33"):
                added_imgs[33] = gr.Image(height = 75, width = 75, min_width = 75, show_label = False, container = True, show_download_button = False, interactive = False)
              with gr.Row(elem_classes = "panelm", elem_id = "row34"):
                added_imgs[34] = gr.Image(height = 75, width = 75, min_width = 75, show_label = False, container = True, show_download_button = False, interactive = False)
              with gr.Row(elem_classes = "panelm", elem_id = "row35"):
                added_imgs[35] = gr.Image(height = 75, width = 75, min_width = 75, show_label = False, container = True, show_download_button = False, interactive = False)
            with gr.Row(min_width=550):
              with gr.Row(elem_classes = "panelm", elem_id = "row36"):
                added_imgs[36] = gr.Image(height = 75, width = 75, min_width = 75, show_label = False, container = True, show_download_button = False, interactive = False)
              with gr.Row(elem_classes = "panelm", elem_id = "row37"):
                added_imgs[37] = gr.Image(height = 75, width = 75, min_width = 75, show_label = False, container = True, show_download_button = False, interactive = False)
              with gr.Row(elem_classes = "panelm", elem_id = "row38"):
                added_imgs[38] = gr.Image(height = 75, width = 75, min_width = 75, show_label = False, container = True, show_download_button = False, interactive = False)
              with gr.Row(elem_classes = "panelm", elem_id = "row39"):
                added_imgs[39] = gr.Image(height = 75, width = 75, min_width = 75, show_label = False, container = True, show_download_button = False, interactive = False)
              with gr.Row(elem_classes = "panelm", elem_id = "row40"):
                added_imgs[40] = gr.Image(height = 75, width = 75, min_width = 75, show_label = False, container = True, show_download_button = False, interactive = False)
              with gr.Row(elem_classes = "panelm", elem_id = "row41"):
                added_imgs[41] = gr.Image(height = 75, width = 75, min_width = 75, show_label = False, container = True, show_download_button = False, interactive = False)
            with gr.Row(min_width=550):
              with gr.Row(elem_classes = "panelm", elem_id = "row42"):
                added_imgs[42] = gr.Image(height = 75, width = 75, min_width = 75, show_label = False, container = True, show_download_button = False, interactive = False)
              with gr.Row(elem_classes = "panelm", elem_id = "row43"):
                added_imgs[43] = gr.Image(height = 75, width = 75, min_width = 75, show_label = False, container = True, show_download_button = False, interactive = False)
              with gr.Row(elem_classes = "panelm", elem_id = "row44"):
                added_imgs[44] = gr.Image(height = 75, width = 75, min_width = 75, show_label = False, container = True, show_download_button = False, interactive = False)
              with gr.Row(elem_classes = "panelm", elem_id = "row45"):
                added_imgs[45] = gr.Image(height = 75, width = 75, min_width = 75, show_label = False, container = True, show_download_button = False, interactive = False)
              with gr.Row(elem_classes = "panelm", elem_id = "row46"):
                added_imgs[46] = gr.Image(height = 75, width = 75, min_width = 75, show_label = False, container = True, show_download_button = False, interactive = False)
              with gr.Row(elem_classes = "panelm", elem_id = "row47"):
                added_imgs[47] = gr.Image(height = 75, width = 75, min_width = 75, show_label = False, container = True, show_download_button = False, interactive = False)            
            #image_selected = gr.Image(height = 1000, wdth = 550, show_label = False, container = True, show_download_button = False)
          with gr.Column(scale = 1):
            multiCaption = gr. Button(value = "Multi Image Detect Captions", variant="primary", scale = 1, size="sm", visible = False, elem_id="multiCaption")
            multiDetect = gr. Button(value = "Multi Image Detect Objects", variant="primary", scale = 1, size="sm", visible = False, elem_id="multiDetect")
            caption = gr. Button(value = "Detect Captions", variant="primary", scale = 1, size="sm", visible = False, elem_id="caption")
            detect = gr. Button(value = "Detect Objects", variant="primary", scale = 1, size="sm", visible = False, elem_id="detect")
            addAI = gr. Button(value = "Add all", variant="primary", scale = 1, size="sm", elem_id="addAIToDB")
            add = gr.Button(value="Add Selected", variant="primary", scale = 1, size="sm", elem_id="addToDB")                                              
        with gr.Column(elem_classes = "panel4"):
          with gr.Row():
                with gr.Row(scale = 1):
                        gr.Label(value="Import from a folder", show_label=False, container = False, elem_id = "iff")              
          text1 = gr.Textbox(placeholder = "Insert folder location", max_lines = 1,  scale = 1, show_label=False , elem_id = "ifl")
          with gr.Row(elem_id = "importFR"):
             importF =  gr.Button(value="Import", variant="primary", scale = 1, size="sm", elem_id = "importF", elem_classes = "import0")
          importFDone = gr.Label(value="", show_label=False, container = False, elem_id = "id1")
          gr.Label(value="Select Detection Model", show_label=False, container = False, elem_id = "sdm")
          sdm = gr.Dropdown(["yolov3u.pt", "yolov5x6u.pt", "yolov8x.pt"], value = "yolov3u.pt", show_label=False, container = False, elem_id = "ddd")
          gr.Label(value="Select Caption Model", show_label=False, container = False, elem_id = "scm")
          scm = gr.Dropdown(["llava-1.5-7b-hf", "llava-v1.6-mistral-7b-hf", "vip-llava-7b-hf"], value = "llava-1.5-7b-hf", show_label=False, container = False, elem_id = "cdd")
          with gr.Column(scale=1):
                pass 
          with gr.Column(scale=1):
                pass 
          gr.Label(value="WEBUI EXTENSION", show_label=False, container = False, elem_id = "lwe")
          with gr.Column(elem_classes = "panel5"):
            with gr.Row():
              gr.Label(value="processing", show_label=False, container = False, elem_id = "process")
              with gr.Row(scale=1):
                pass 
              with gr.Row(scale=1):
                pass
            with gr.Row():
               gr.Label(value="Detection", show_label=False, container = False, elem_id = "det")
            label1 = gr.Label(value="", show_label=False, elem_id = "lbl1")
            with gr.Row():
               gr.Label(value="Caption", show_label=False, container = False, elem_id = "cap")
            label2 = gr.Label(value="", show_label=False, elem_id = "lbl2")                   
        with gr.Column():
          with gr.Column(elem_classes = "panel2"):
            with gr.Column(scale = 1):  
              with gr.Row():
                  with gr.Row(scale = 1, elem_id = "rowIfs"):
                          gr.Label(value="Import from another source", show_label=False, container = False, elem_id = "ifas")   
              text2 = gr.Textbox(placeholder = "Insert URL", max_lines = 1, show_label=False, elem_id = "iu")
              with gr.Row(elem_id = "importUR"):
                 importU = gr.Button(value="Import", variant="primary", size="sm", elem_classes = "import0", elem_id = "importU")  
              importUDone = gr.Label(value="", show_label=False, container = False, elem_id = "id2")
              #unusedLabel = gr.Label(value="a", show_label=False, scale = 1, container = False, elem_id = "id3")     
              # with gr.Column(scale = 1): 
              #   with gr.Column():
              #       pass
              #   with gr.Column():   
              #     importU = gr.Button(value="Import", variant="primary", size="sm",elem_classes = "import0", elem_id = "importU")
              # with gr.Column(scale = 2): 
              #   with gr.Column():
              #       pass
              #   with gr.Column():   
              #     importUDone = gr.Label(value="", show_label=False, container = False, elem_id = "id2")    
                                 
          with gr.Column(elem_classes = "panel2"):
            with gr.Column(scale = 1): 
              with gr.Row():
                with gr.Row(scale=1, elem_id = "rowDets"):                
                        gr.Label(value="Detection Settings", show_label=False, container = False, elem_id = "dets")
                with gr.Row(scale=1):
                        pass
              with gr.Row():
                gr.Label(value="Minimum capture width:", show_label=False, container = False, elem_id = "mcw")
                with gr.Row(scale=1):
                  with gr.Column():
                    with gr.Column():
                      pass
                    with gr.Column():
                      mcw = gr.Number(value=objWidth, precision=0, minimum=0, interactive = True, show_label=False, container = False, elem_id = "mcwn")
                # with gr.Row(scale=1):
                #         pass            
              with gr.Row():
                gr.Label(value="Minimum capture height:", show_label=False, container = False, elem_id = "mch")
                with gr.Row(scale=1):
                  with gr.Column():
                    with gr.Column():
                      pass
                    with gr.Column():
                     mch = gr.Number(value=objHeight, precision=0, minimum=0, interactive = True, show_label=False, container = False, elem_id = "mchn")   
                # with gr.Row(scale=1):
                #         pass         
              with gr.Row():
                gr.Label(value="Minimum confidence level:", show_label=False, container = False, elem_id = "mcl")
                with gr.Row(scale=1):
                  with gr.Column():
                   with gr.Column():
                      pass
                   with gr.Column():
                     mcfl = gr.Slider(value=confl, minimum=0.00, maximum=1.00, step = 0.01, scale = 1, interactive = True, show_label=False, container = False, elem_id = "mcln")
                # with gr.Row(scale=1):
                #         pass                           
              with gr.Row(elem_id = "ss1"):
                        reset1 = gr.Button(value = "Reset", size="sm", min_width = 30, elem_id = "reset1")
                        save1 = gr.Button(value = "Save", size="sm", min_width = 30, elem_id = "save1")
            objsv = gr.Label(value="", show_label=False, container = False)  
    #with gr.Group():  
          with gr.Column(elem_classes = "panel2"):
            with gr.Row():
                with gr.Row(scale=1, elem_id = "rowCaps"):           
                        gr.Label(value="Caption Settings", show_label=False, container = False, elem_id = "caps")
                with gr.Row(scale=1):
                        pass

            with gr.Row():
                gr.Label(value="Minimum caption length:", scale = 1, show_label=False, container = False, elem_id = "mcpl")
                with gr.Row(scale=1):
                  with gr.Column():
                   with gr.Column():
                      pass
                   with gr.Column():    
                     mcpl = gr.Number(value=capLen, precision=0, minimum=0, scale = 1, min_width = 50, interactive = True, show_label=False, container = False, elem_id = "mcpln")
                   with gr.Column():
                      pass  
                # with gr.Row(scale=1):
                #         pass         

            with gr.Row(elem_id = "ss2"):
                        reset2 = gr.Button(value = "Reset", size="sm", min_width = 30, elem_id = "reset2")
                        save2 = gr.Button(value = "Save", size="sm", min_width = 30, elem_id = "save2")
            capsv = gr.Label(value="", show_label=False, container = False)               
      caption.click(cap_detection,  inputs=[scm],  outputs=[label2]).then(None, None, None, _js = "enabled_buttons")
      detect.click(img_detection,  inputs=[sdm], outputs=[label1])
      multiCaption.click(multi_img_caption, inputs=[scm], outputs=[label2]).then(reset_images, None, outputs = [image] + [added_imgs[i] for i in range(24, 48)]).then(None, None, None, _js = "enabled_buttons")
      multiDetect.click(multi_img_detection, inputs=[sdm], outputs=[label1])
      add.click(no_fn, None, None, _js = "disable_buttons").then(no_fn, None, None, _js = "AddToDataBase")
      addAI.click(no_fn, None, None, _js = "disable_buttons").then(no_fn, None, None, _js = "AddMultiToDataBase")
      label1.change(label_rest, outputs = [label1])
      label2.change(label_rest, outputs = [label2])  
      importF.click(no_fn, None, None, _js = "disable_buttons").then(update_images, inputs=[text1], outputs = [image] + [added_imgs[i] for i in range(0, 48)] +  [importFDone]).then(None, None, None, _js = "enabled_buttons")
      importU.click(no_fn, None, None, _js = "disable_buttons").then(import_from_url, inputs=[text2], outputs = [image] + [added_imgs[i] for i in range(0, 48)] +  [importUDone]).then(None, None, None, _js = "enabled_buttons")
      less.click(no_fn, None, None, _js = "disable_buttons").then(minus, outputs = [image]).then(None, None, None, _js = "enabled_buttons")
      greater.click(no_fn, None, None, _js = "disable_buttons").then(plus, outputs = [image]).then(None, None, None, _js = "enabled_buttons")
      addImg.click(no_fn, None, None, _js = "disable_buttons").then(add_image, inputs = None, outputs = [image]+[added_imgs[i] for i in range(0, 48)]).then(None, None, None, _js = "enabled_buttons")
      removeImg.click(no_fn, None, None, _js = "disable_buttons").then(remove_image, inputs = None, outputs = [image]+[added_imgs[i] for i in range(0, 48)]).then(None, None, None, _js = "enabled_buttons")
      reset1.click(no_fn, None, None, _js = "disable_buttons").then(reset_objsettings, outputs = [mcw, mch, mcfl]).then(None, None, None, _js = "enabled_buttons")
      reset2.click(no_fn, None, None, _js = "disable_buttons").then(reset_capsettings, outputs = [mcpl]).then(None, None, None, _js = "enabled_buttons")
      save1.click(no_fn, None, None, _js = "disable_buttons").then(save_settings, inputs=[mcw, mch, mcfl, mcpl], outputs = [objsv]).then(None, None, None, _js = "enabled_buttons")
      save2.click(no_fn, None, None, _js = "disable_buttons").then(save_settings, inputs=[mcw, mch, mcfl, mcpl], outputs = [capsv]).then(None, None, None, _js = "enabled_buttons")
      objsv.change(label_rest, outputs = [objsv])
      capsv.change(label_rest, outputs = [capsv])
      importFDone.change(label_rest, outputs = [importFDone])
      importUDone.change(label_rest, outputs = [importUDone])
      for i in range(0, 48):
         added_imgs[i].select(img_clicked, inputs = [gr.Number(value=i, precision=0, visible=False)], outputs = [image])       
  return [(Objects_Detections_and_Captions, "Objects Detections and Captions", "Objects_Detections_and_Captions")]

script_callbacks.on_ui_tabs(on_ui_tabs)