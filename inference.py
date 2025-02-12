
import random
import numpy as np
import torch
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from transformers import AutoModel, AutoTokenizer
import opencc
from ultralytics import YOLO
from config.configu import *
from utils.utils import *
import logging
import argparse

def setup_logger(log_file):
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')
    logger = logging.getLogger()
    return logger

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
   
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 

cc = opencc.OpenCC('t2s.json')
set_seed(SEED)
converter_t2s = opencc.OpenCC('t2s')


def single_rec(model,tokenizer,detect_model,generation_config,image_path,prompt,use_p,hard_vq,drop_zero,repetition_penalty,verbose):
    response, history = model.chat_ocr(tokenizer, detect_model,image_path, prompt, generation_config,
                            use_p=use_p,
                            hard_vq=hard_vq,
                            drop_zero=drop_zero,repetition_penalty=repetition_penalty,return_history=True,verbose=verbose)
    print(f'User: {prompt}\nAssistant: {response}')

def folder_rec(model,tokenizer,detect_model,generation_config,folder_path,prompt,save_name,use_p,hard_vq,drop_zero,repetition_penalty,verbose):
    results=[]

    all_images=get_image_paths(folder_path)
    for pic in tqdm(all_images):
        pic_path=os.path.join(folder_path,pic)
        try:
            response, history = model.chat_ocr(tokenizer, detect_model,pic_path, prompt, generation_config,
                                use_p=use_p,
                                hard_vq=hard_vq,
                                drop_zero=drop_zero,repetition_penalty=repetition_penalty,return_history=True,verbose=verbose)
        except Exception as e:
            print(f"An error has occured:\n{e}")
            response="ERROR!"
        print(f'User: {prompt}\nAssistant: {response}')
        results.append({"imagePath":pic_path,'prompt':prompt,'response':response})
    if not save_name.endswith('json'):
        save_name+='_result.json'
    save_json(save_name,results)


def main():
    parser = argparse.ArgumentParser(description="args for inference task")

    parser.add_argument('--tgt', type=str,help='Recognition target')
    parser.add_argument('--prompt', type=str,default='这幅书法作品内容是什么？',help='Prompt for recognition')
    parser.add_argument('--save_name',type=str,default="recognition.json",help="Storage of results if multiple images recognition mode")

    parser.add_argument('--use_p', type=bool, default=True,help='Decide the usage of perceiver resampler')
    parser.add_argument('--hard_vq', type=bool, default=False,help='Decide the usage of closest similarity match')
    parser.add_argument('--drop_zero', type=bool, default=False,help='Decide the deletion of zero padding in pseudo tokens')
    parser.add_argument('--verbose', type=bool, default=False,help='Decide the output of extra information')
    parser.add_argument('--repetition_penalty', type=float, default=1.0,help='Repetition penalty for generation')
    

    args = parser.parse_args()

    if not isinstance(args.tgt,str):
        raise ValueError(f"The target should a string, not a instance of {type(args.tgt)}!")

    
    model = AutoModel.from_pretrained(
            INTERNVL_PATH,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(INTERNVL_PATH, trust_remote_code=True)

    generation_config = dict(
            num_beams=1,
            max_new_tokens=1024,
            do_sample=False,
        )

    detect_model=YOLO(YOLO_CHECKPOINT)
    if is_image(args.tgt):
        print("Single image recognition mode.")
        single_rec(
                    model,
                    tokenizer,
                    detect_model,
                    generation_config,
                    args.tgt,
                    args.prompt,
                    args.use_p,
                    args.hard_vq,
                    args.drop_zero,
                    args.repetition_penalty,
                    args.verbose)
    elif os.path.isdir(args.tgt):
        print("Multiple images recognition mode")
        os.makedirs('results',exist_ok=True)
        folder_rec(
                   model,
                   tokenizer,
                   detect_model,
                   generation_config,
                   args.tgt,
                   args.prompt,
                   os.path.join('results',args.save_name),
                   args.use_p,
                   args.hard_vq,
                   args.drop_zero,
                   args.repetition_penalty,
                   args.verbose)
    else:
        raise ValueError(f"The target should be either a image path or a folder that contain images!")

def single_image_wrapped(image,prompts):
    model = AutoModel.from_pretrained(
            INTERNVL_PATH,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(INTERNVL_PATH, trust_remote_code=True)

    generation_config = dict(
            num_beams=1,
            max_new_tokens=1024,
            do_sample=False,
        )
    detect_model=YOLO(YOLO_CHECKPOINT)


    temp_dir = "temp_images"
    os.makedirs(temp_dir, exist_ok=True)
    
    # 获取图片的文件名，并保存
    temp_image_path = os.path.join(temp_dir, "uploaded_image.png")
    image.save(temp_image_path)
    single_rec(
                    model,
                    tokenizer,
                    detect_model,
                    generation_config,
                    temp_image_path,
                    prompts,
                    True,
                    False,
                    True,
                    1.2,
                    False)
if __name__=='__main__':

    main()


