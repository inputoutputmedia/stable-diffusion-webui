from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig, LlavaNextProcessor, LlavaNextForConditionalGeneration, VipLlavaForConditionalGeneration

VipLlavaForConditionalGeneration.from_pretrained("llava-hf/vip-llava-7b-hf", cache_dir = "C:/decentre/appdata"+"/models/caption")
LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", cache_dir = "C:/decentre/appdata"+"/models/caption")
LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf", cache_dir = "C:/decentre/appdata"+"/models/caption")  