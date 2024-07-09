from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig, LlavaNextProcessor, LlavaNextForConditionalGeneration, VipLlavaForConditionalGeneration
import os
import torch
import gc

Model1 = None
Model2 = None
Model3 = None
processor1 = None
processor2 = None
processor3 = None

captionFolder = "C:/decentre/appdata/models/caption"
isExist = os.path.exists(captionFolder)
if not isExist:
    # Create a new directory because it does not exist
     os.makedirs(captionFolder)

isExist = os.path.exists("C:/decentre/appdata/models/caption/models--llava-hf--vip-llava-7b-hf")
if not isExist:
    Model1 = VipLlavaForConditionalGeneration.from_pretrained("llava-hf/vip-llava-7b-hf", cache_dir = "C:/decentre/appdata"+"/models/caption")
    processor1 = AutoProcessor.from_pretrained("llava-hf/vip-llava-7b-hf", cache_dir = "C:/decentre/appdata"+"/models/caption")

isExist = os.path.exists("C:/decentre/appdata/models/caption/models--llava-hf--llava-v1.6-mistral-7b-hf")
if not isExist:
    Model2 = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", cache_dir = "C:/decentre/appdata"+"/models/caption")
    processor2 = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", cache_dir = "C:/decentre/appdata"+"/models/caption")

isExist = os.path.exists("C:/decentre/appdata/models/caption/models--llava-hf--llava-1.5-7b-hf")
if not isExist:
    Model3 = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf", cache_dir = "C:/decentre/appdata"+"/models/caption")
    processor3 = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf", cache_dir = "C:/decentre/appdata"+"/models/caption")



del Model1
del Model2
del Model3
del processor1
del processor2
del processor3

gc.collect()
torch.cuda.empty_cache()
     