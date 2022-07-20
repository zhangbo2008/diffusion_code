# !pip install diffusers #   注意要使用 python3.9才行.
from diffusers import DiffusionPipeline
import PIL.Image
import numpy as np

model_id = "google/ddpm-cifar10"
import torch
# load model and scheduler
ddpm = DiffusionPipeline.from_pretrained(model_id)
#============去噪

# import cv2
# old=cv2.imread('airplane10zaoyin.png')
old=PIL.Image.open('airplane10zaoyin.png')
old=np.array(old).reshape([1,4,32,32])
old=old[0,:3,:,:]




old=torch.tensor(old)
old=old/127.5-1
# run pipeline in inference (sample random noise and denoise)
image = ddpm.call2(image=old)

# process image to PIL
image_processed = image.cpu().permute(0, 2, 3, 1)
image_processed = (image_processed + 1.0) * 127.5
image_processed = image_processed.numpy().astype(np.uint8)
image_pil = PIL.Image.fromarray(image_processed[0])

# save image
image_pil.save("test2.png")
