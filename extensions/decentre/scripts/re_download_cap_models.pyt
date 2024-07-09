from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig, LlavaNextProcessor, LlavaNextForConditionalGeneration, VipLlavaForConditionalGeneration
import os
import torch
import gc

Model1 = VipLlavaForConditionalGeneration.from_pretrained("llava-hf/vip-llava-7b-hf", cache_dir = "C:/decentre/appdata"+"/models/caption")
processor1 = AutoProcessor.from_pretrained("llava-hf/vip-llava-7b-hf", cache_dir = "C:/decentre/appdata"+"/models/caption")
del Model1
del processor1
gc.collect()
torch.cuda.empty_cache()


Model2 = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", cache_dir = "C:/decentre/appdata"+"/models/caption")
processor2 = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", cache_dir = "C:/decentre/appdata"+"/models/caption")
del Model2
del processor2
gc.collect()
torch.cuda.empty_cache()

Model3 = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf", cache_dir = "C:/decentre/appdata"+"/models/caption")
processor3 = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf", cache_dir = "C:/decentre/appdata"+"/models/caption")
del Model3
del processor3
gc.collect()
torch.cuda.empty_cache()

print("caps redownloaded")