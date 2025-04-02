import os
import numpy as np
import torch
from PIL import Image

parent_folder = 'path/where/HR'
target_scale = 0.25
output_folder = 'path/where/LR'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# bicubic downsampling 
def bicubic_downsample(image, scale):
    width, height = image.size
    new_width = int(width * scale)
    new_height = int(height * scale)
    return image.resize((new_width, new_height), Image.BICUBIC)



for f in os.listdir(parent_folder):
    f_path = os.path.join(parent_folder, f)
    if not os.path.isdir(f_path):
        continue
    bicubic_path = os.path.join(output_folder, f)
    
    if not os.path.exists(bicubic_path):
        os.makedirs(bicubic_path)
    for frame in os.listdir(os.path.join(parent_folder, f)):
        frame_path = os.path.join(f_path, frame) 
        if frame.endswith((".jpg", ".jpeg", ".png")):
            bicubic_save_path = os.path.join(bicubic_path, frame)
            print(bicubic_save_path)
            img = Image.open(frame_path)
            downsampled_img = bicubic_downsample(img, target_scale)
            downsampled_img.save(bicubic_save_path)
            downsampled_img.close()

