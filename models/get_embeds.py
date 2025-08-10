# import os; 
import sys
sys.path.append('/home/luoyx/InternVL/InternVL-main_test')
import warnings
from typing import Any, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.optim as optim
import random
from einops import rearrange, repeat
from torch import einsum
import transformers
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer, AutoConfig, ViTModel
from transformers import (AutoModel, GenerationConfig, LlamaForCausalLM,
                          LlamaTokenizer)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput, logging

from modeling_intern_vit import InternVisionModel
from PIL import Image
import cv2
import numpy as np
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SCALE_DOWN_RANGE = (0.4, 0.8)  # 缩小比例
SCALE_UP_RANGE = (1.2, 2.0)    # 放大比例

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def build_transform_cv(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.ToTensor(),
        T.Resize((input_size, input_size), interpolation=T.InterpolationMode.BICUBIC),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')

    # resize图片
    image = image.resize((448, 448))
    
    transform = build_transform(input_size=input_size)
    pixel_values = transform(image).unsqueeze(0)
    return pixel_values

def load_image_cv(image_file, input_size=448, max_num=12):
    # 使用OpenCV读取图像
    if isinstance(image_file,str):
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if isinstance(image_file,Image.Image):
        image=np.array(image_file)
        
    

    # # resize图片
    # image = cv2.resize(image, (448, 448), interpolation=cv2.INTER_CUBIC)
    
    transform = build_transform_cv(input_size=input_size)
    pixel_values = transform(image)#.unsqueeze(0)
 
    return pixel_values

def load_image_cv_with_aug(image, input_size=448):
    # 使用OpenCV读取图像
    if isinstance(image,str):
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
    if isinstance(image,Image.Image):
        image=np.array(image)

    # 获取图像的原始宽度和高度
    h, w, _ = image.shape
    short_edge = min(h, w)

    # 如果图像的短边大于input_size，随机缩小再放大回来
    if short_edge > input_size:
        # 随机取缩小比例
        scale_factor = random.uniform(*SCALE_DOWN_RANGE)
        new_size = int(short_edge * scale_factor)
        image = cv2.resize(image, (new_size, new_size), interpolation=cv2.INTER_AREA)
        
        # 放大回接近原图大小
        scale_up_factor = random.uniform(*SCALE_UP_RANGE)
        resized_size = int(new_size * scale_up_factor)
        image = cv2.resize(image, (resized_size, resized_size), interpolation=cv2.INTER_LINEAR)

    # 如果图像的短边小于input_size，随机放大再缩小回来
    elif short_edge < input_size:
        # 随机取放大比例
        scale_factor = random.uniform(*SCALE_UP_RANGE)
        new_size = int(short_edge * scale_factor)
        image = cv2.resize(image, (new_size, new_size), interpolation=cv2.INTER_LINEAR)
        
        # 缩小回接近原图大小
        scale_down_factor = random.uniform(*SCALE_DOWN_RANGE)
        resized_size = int(new_size * scale_down_factor)
        image = cv2.resize(image, (resized_size, resized_size), interpolation=cv2.INTER_AREA)

    # 最后，将图像resize到指定的 (input_size, input_size)
    transform = build_transform_cv(input_size=input_size)
    pixel_values = transform(image)  # 将图像转换为张量，并进行其他预处理操作
    return pixel_values

def pixel_shuffle(x, scale_factor=0.5):
    n, w, h, c = x.size()
    # N, W, H, C --> N, W, H * scale, C // scale
    x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
    # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
    x = x.permute(0, 2, 1, 3).contiguous()
    # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
    x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                int(c / (scale_factor * scale_factor)))
    x = x.permute(0, 2, 1, 3).contiguous()
    return x

import torch.nn.functional as F

def extract_embeds(tokenizer, vision_model, mlp1, downsample_ratio, tok_embedding, pixel_values, questions, test_print=False):
    # modified batch-wise?
    # 批量处理questions
    tokenizer.pad_token_id = 0
    model_inputs = tokenizer(questions, return_tensors='pt', padding='max_length', max_length=4, truncation=True).to(pixel_values.device)

    input_ids = model_inputs['input_ids'][:, 1:]  # 去掉每个序列的第一个token (b, 3)
    #print(model_inputs['input_ids'])
    # 处理图像嵌入
    vit_embeds = vision_model(
        pixel_values=pixel_values,
        output_hidden_states=False,
        return_dict=True
    ).last_hidden_state[ :, 1:, :]  # 去掉[CLS] token (b, 1025-1, 1024)

    # 调整形状
    h = w = int(vit_embeds.shape[1] ** 0.5)
    vit_embeds = vit_embeds.view(vit_embeds.shape[0], h, w, -1)

    # 下采样
    vit_embeds = pixel_shuffle(vit_embeds, scale_factor=downsample_ratio)
    vit_embeds = vit_embeds.view(vit_embeds.shape[0], -1, vit_embeds.shape[-1])

    # # 通过 MLP 处理
    vit_embeds = mlp1(vit_embeds)

    # 生成输入嵌入
    input_embeds = tok_embedding(input_ids)
    if test_print:
        return vit_embeds, input_embeds, input_ids
