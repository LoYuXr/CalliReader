
import random
import numpy as np
import torch
from PIL import Image
from io import BytesIO
from typing import List, Dict, Tuple, Any
import pandas as pd
import re

Image.MAX_IMAGE_PIXELS = None
import Levenshtein
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

def remove_chinese_punctuation(text):
    chinese_punctuation_regex = re.compile(r'[\u3002\uFF1F\uFF01\u3001\uff0c\u300c\u300d\u300e\u300f\u2018\u2019\u201c\u201d\u2013\u2014\u2026\u3010\u3011\u300a\u300b\uff1a\uff1b]')
    return chinese_punctuation_regex.sub('', text)

def remove_english_punctuation(text):  
    english_punctuation_regex = re.compile(r'[,\.!?:\'";\(\)\[\]\{\}\-\n\*1234567890]') 
    return english_punctuation_regex.sub('', text) 

def get_clean_string(str):
    return remove_chinese_punctuation(remove_english_punctuation(str))

def get_parquet(parquet_path: str) -> Tuple[List[Image.Image], List[Dict]]:
    df = pd.read_parquet(parquet_path)
    
    list_of_images = []
    list_of_annotations = []
    
    
    for index, row in df.iterrows():
        try:
            # 解析标签信息
            labels = json.loads(row['annotation'])
            image_bytes = row['image']['bytes']
            
            image = Image.open(BytesIO(image_bytes))
            
            list_of_images.append(image)
            list_of_annotations.append(labels)
            
        except Exception as e:
            print(f"Row {index} Error: {e}")
            continue
    
    return list_of_images, list_of_annotations


def evaluate_accuracy(responses, correct_answers):
    """
    评估选择题回答准确率
    
    参数:
    - responses: 模型回答列表
    - correct_answers: 正确答案列表，每个元素为 (正确选项, 正确文本, 错误文本1, 错误文本2)
    
    返回:
    - accuracy: 准确率百分比
    """
    assert len(responses) == len(correct_answers), "Responses and answers must have the same length."
        
    correct_count = 0

    for response, correct_answer in zip(responses, correct_answers):
        # 检查是否包含选项字母
        contains_A = 'A' in response
        contains_B = 'B' in response
        contains_C = 'C' in response
        
        # 检查是否包含选项文本内容
        contain_text_gt = correct_answer[1] in response  # 正确选项文本
        contain_wrong_0 = correct_answer[2] in response  # 错误选项1文本
        contain_wrong_1 = correct_answer[3] in response  # 错误选项2文本
            
        # 规则1: 如果包含多个选项字母，判为错误
        if sum([contains_A, contains_B, contains_C]) > 1:
            is_correct = False
        else:
            # 规则2: 根据选择的选项字母判断
            chosen_option = 'A' if contains_A else 'B' if contains_B else 'C' if contains_C else None
            is_correct = (chosen_option == correct_answer[0])

        # 规则3: 如果包含正确文本但不包含错误文本，判为正确；如果同时包含正确和错误文本，判为错误
        if contain_text_gt:
            if contain_wrong_0 or contain_wrong_1:
                is_correct = False
            else:
                is_correct = True
            
        if is_correct:
            correct_count += 1
    
    accuracy = correct_count / len(responses) * 100
    return accuracy

def single_rec(model,tokenizer,detect_model,generation_config,image_path,prompt,use_p,hard_vq,drop_zero,repetition_penalty,verbose):
    response, history = model.chat_ocr(tokenizer, detect_model,image_path, prompt, generation_config,
                            use_p=use_p,
                            hard_vq=hard_vq,
                            drop_zero=drop_zero,repetition_penalty=repetition_penalty,return_history=True,verbose=verbose)
    return cc.convert(response)

def test_full_page(parquet_path,save_json_path,model,tokenizer,detect_model,generation_config,prompt,use_p,hard_vq,drop_zero,repetition_penalty,verbose):
    images,annotations=get_parquet(parquet_path)
    to_be_save={"detailed":[]}
    avg_pre=0
    avg_rec=0
    avg_f1=0
    avg_ned=0
    count=0
    for img,annot in zip(tqdm(images),annotations):
        response=single_rec(model,tokenizer,detect_model,generation_config,img,prompt,use_p,hard_vq,drop_zero,repetition_penalty,verbose)
        gt=get_clean_string(annot['reference'])
        response=list(response)
        gt=list(gt)


        precision, recall, f1_score = calculate_metrics(response,gt)
        distance = Levenshtein.distance(response, gt)
        max_len = max(len(response), len(gt))
        ned = distance / max_len 
        to_be_save['detailed'].append({"imgPath":annot['imagePath'],"prompt":prompt,"output":''.join(response),"gt":''.join(gt), "precision":precision,"recall":recall,"f1":f1_score,"ned":ned})

        avg_pre+=precision
        avg_rec+=recall
        avg_f1+=f1_score
        avg_ned+=ned
        count+=1
    
    avg_pre/=count
    avg_rec/=count
    avg_f1/=count
    avg_ned/=count
    to_be_save['average']={"ave_precison":avg_pre,
                           "avg_recall":avg_rec,
                           "avg_f1":avg_f1,
                           "avg_ned":avg_ned,
                          }
    with open(save_json_path, 'w', encoding='utf-8') as f:
        json.dump(to_be_save, f, ensure_ascii=False, indent=4)

def test_region_wise(parquet_path,save_json_path,model,tokenizer,detect_model,generation_config,prompt,use_p,hard_vq,drop_zero,repetition_penalty,verbose):
    images,annotations=get_parquet(parquet_path)
    to_be_save={"detailed":[]}
    avg_pre=0
    avg_rec=0
    avg_f1=0
    avg_ned=0
    count=0
    for img,annot in zip(tqdm(images),annotations):
        [x1,y1],[x2,y2]=annot['region']
        img=np.array(img.convert("RGB"))
        sub_img=Image.fromarray(img[y1:y2,x1:x2])

        response=single_rec(model,tokenizer,detect_model,generation_config,sub_img,prompt,use_p,hard_vq,drop_zero,repetition_penalty,verbose)
        gt=get_clean_string(annot['answer'])
        response=list(response)
        gt=list(gt)

        precision, recall, f1_score = calculate_metrics(response,gt)
        distance = Levenshtein.distance(response, gt)
        max_len = max(len(response), len(gt))
        ned = distance / max_len 
        to_be_save['detailed'].append({"imgPath":annot['imagePath'],"prompt":prompt,"output":''.join(response),"gt":''.join(gt), "precision":precision,"recall":recall,"f1":f1_score,"ned":ned})

        avg_pre+=precision
        avg_rec+=recall
        avg_f1+=f1_score
        avg_ned+=ned
        count+=1
    
    avg_pre/=count
    avg_rec/=count
    avg_f1/=count
    avg_ned/=count
    to_be_save['average']={"ave_precison":avg_pre,
                           "avg_recall":avg_rec,
                           "avg_f1":avg_f1,
                           "avg_ned":avg_ned,
                          }
    with open(save_json_path, 'w', encoding='utf-8') as f:
        json.dump(to_be_save, f, ensure_ascii=False, indent=4)


def test_choice(parquet_path, save_json_path, model, tokenizer, detect_model, generation_config, use_p=True, hard_vq=False, drop_zero=True, repetition_penalty=1.0, verbose=False):

    # 加载数据
    images, annotations = get_parquet(parquet_path)
    
    # 准备正确答案列表和结果存储
    gt_list = []
    response_list = []
    to_be_save = {
        "detailed": [],
        "summary": {}
    }
    
    # 预处理正确答案信息
    for item in annotations:
        prompt = item["conversations"][0]["value"]
        gt = item["conversations"][1]["value"]  # 正确选项字母
        
        # 解析选项内容
        lines = prompt.split('\n')
        wrong_line_0 = None
        wrong_line_1 = None
        
        for line in lines:
            if 'A' in line or 'B' in line or 'C' in line:
                if line.startswith(gt + ":"):
                    options_line = line
                else:
                    if wrong_line_0 is None:
                        wrong_line_0 = line
                    elif wrong_line_1 is None:
                        wrong_line_1 = line
        
        # 提取选项文本内容
        text_gt = options_line.split(":")[1].strip()  # 正确选项文本
        wrong_text_0 = wrong_line_0.split(":")[1].strip()  # 错误选项1文本
        wrong_text_1 = wrong_line_1.split(":")[1].strip()  # 错误选项2文本
        
        gt_list.append((gt, text_gt, wrong_text_0, wrong_text_1))
    

    for img,annot,gt_info in zip(tqdm(images[:3]),annotations[:3],gt_list[:3]):

                # 第一轮对话：内容识别
        question = '这幅书法作品内容是什么？'
        response, history = model.chat_ocr(
                    tokenizer, detect_model, img, question, generation_config,
                    use_p=use_p,
                    hard_vq=hard_vq,
                    drop_zero=drop_zero,
                    repetition_penalty=repetition_penalty,
                    return_history=True,
                    verbose=verbose
        )
           

        prompt = annot["conversations"][0]["value"].replace("<image>\n", "")

        match = re.search(r"^(.*?)\n[A-Z]:", prompt, re.DOTALL)
 
        question = prompt + f"\n只需要输出问题的答案，禁止输出其他内容！答案："
            
        response, history = model.chat_ocr(
                tokenizer, detect_model, img, question, generation_config,
                use_p=use_p,
                hard_vq=hard_vq,
                drop_zero=drop_zero,
                repetition_penalty=repetition_penalty,
                history=history,
                return_history=True,
                verbose=verbose
        )
 
     
        
        response_list.append(response)
        
        # 记录详细结果
        to_be_save["detailed"].append({
            "imgPath": annot['image'],
            "output": response,
            "reference": gt_info[0],  # 正确选项字母
        })
    
    # 计算准确率
    accuracy = evaluate_accuracy(response_list, gt_list[:3])
    
    # 保存汇总结果
    to_be_save["summary"] = {
        "total_samples": len(response_list),
        "accuracy": accuracy,
    }
    
    # 保存结果到文件
    with open(save_json_path, 'w', encoding='utf-8') as f:
        json.dump(to_be_save, f, ensure_ascii=False, indent=4)

    return accuracy, to_be_save



def test_bilingual(parquet_path,save_json_path,model,tokenizer,detect_model,generation_config,use_p,hard_vq,drop_zero,repetition_penalty,verbose):
    images,annotations=get_parquet(parquet_path)
    to_be_save={"detailed":[]}
 
    for img,annot in zip(tqdm(images),annotations):

        question = '这幅书法作品内容是什么？'
        response, history = model.chat_ocr(tokenizer, detect_model,img
                                            , question, generation_config,
                                use_p=use_p,
                                hard_vq=hard_vq,
                                drop_zero=drop_zero,repetition_penalty=repetition_penalty,return_history=True,verbose=verbose)
      
        
        prompt = annot["conversations"][0]["value"]
        match = re.search(r"^(.*?)\n[A-Z]:", prompt, re.DOTALL)
        if match:
                substring = match.group(1).strip()
        else:
                substring = prompt

        question = substring 
        response, history = model.chat_ocr(tokenizer, detect_model,img, question, generation_config,
                                  use_p=use_p,
                                hard_vq=hard_vq,
                                drop_zero=drop_zero,repetition_penalty=repetition_penalty,return_history=True,verbose=verbose,history=history)
        reference=annot['conversations'][-1]['value']
        to_be_save['detailed'].append({"imgPath":annot['image'],"output":response.split('ENGLISH:')[-1],"reference":reference})

    
    with open(save_json_path, 'w', encoding='utf-8') as f:
        json.dump(to_be_save, f, ensure_ascii=False, indent=4)


def test_intent(parquet_path,save_json_path,model,tokenizer,detect_model,generation_config,use_p,hard_vq,drop_zero,repetition_penalty,verbose):
    images,annotations=get_parquet(parquet_path)
    to_be_save={"detailed":[]}
 
    for img,annot in zip(tqdm(images),annotations):

        question = '这幅书法作品内容是什么？'
        response, history = model.chat_ocr(tokenizer, detect_model,img
                                            , question, generation_config,
                                use_p=use_p,
                                hard_vq=hard_vq,
                                drop_zero=drop_zero,repetition_penalty=repetition_penalty,return_history=True,verbose=verbose)
      
        
        prompt = annot["conversations"][0]["value"]
        match = re.search(r"^(.*?)\n[A-Z]:", prompt, re.DOTALL)
        if match:
                substring = match.group(1).strip()
        else:
                substring = prompt

        question = substring 
        response, history = model.chat_ocr(tokenizer, detect_model,img, question, generation_config,
                                  use_p=use_p,
                                hard_vq=hard_vq,
                                drop_zero=drop_zero,repetition_penalty=repetition_penalty,return_history=True,verbose=verbose,history=history)
        reference=annot['conversations'][-1]['value']
        to_be_save['detailed'].append({"imgPath":annot['image'],"output":response.split("INTENT:")[-1],"reference":reference})

    
    with open(save_json_path, 'w', encoding='utf-8') as f:
        json.dump(to_be_save, f, ensure_ascii=False, indent=4)


def main():
    parser = argparse.ArgumentParser(description="args for inference task")

    parser.add_argument('--type', type=str, choices=['full_page', 'region_wise', 'choice', 'bilingual', 'intent'], 
                   help='Evaluation Type (full_page, region_wise, choice, bilingual, intent)')
    parser.add_argument('--save_name',type=str,default="exp",help="Storage of results if multiple images recognition mode")
    parser.add_argument('--data', type=str,default='./CalliBench',help='Evaluation Data Directory')

    parser.add_argument('--use_p', type=bool, default=True,help='Decide the usage of perceiver resampler')
    parser.add_argument('--hard_vq', type=bool, default=False,help='Decide the usage of closest similarity match')
    parser.add_argument('--drop_zero', type=bool, default=False,help='Decide the deletion of zero padding in pseudo tokens')
    parser.add_argument('--verbose', type=bool, default=False,help='Decide the output of extra information')
    parser.add_argument('--repetition_penalty', type=float, default=1.0,help='Repetition penalty for generation')
    

    args = parser.parse_args()

    save_dir=f"outputs/{args.save_name}"
    os.makedirs(save_dir,exist_ok=True)
    
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
    if args.type=='full_page':
        prompt="读出图中所有文字。"

        parquet_path=os.path.join(args.data,"full_page_ocr/easy/easy.parquet")
        save_json_path=os.path.join(save_dir,"full_page_easy.json")
        test_full_page(parquet_path,save_json_path,model,tokenizer,detect_model,generation_config,prompt,args.use_p,args.hard_vq,args.drop_zero,args.repetition_penalty,args.verbose)

        parquet_path=os.path.join(args.data,"full_page_ocr/medium/medium.parquet")
        save_json_path=os.path.join(save_dir,"full_page_medium.json")
        test_full_page(parquet_path,save_json_path,model,tokenizer,detect_model,generation_config,prompt,args.use_p,args.hard_vq,args.drop_zero,args.repetition_penalty,args.verbose)
        
        parquet_path=os.path.join(args.data,"full_page_ocr/hard/hard.parquet")
        save_json_path=os.path.join(save_dir,"full_page_hard.json")
        test_full_page(parquet_path,save_json_path,model,tokenizer,detect_model,generation_config,prompt,args.use_p,args.hard_vq,args.drop_zero,args.repetition_penalty,args.verbose)

    if args.type=="region_wise":
        prompt="读出图中区域所有文字。"

        parquet_path=os.path.join(args.data,"region-wise/region.parquet")
        save_json_path=os.path.join(save_dir,"region_wise.json")
        test_region_wise(parquet_path,save_json_path,model,tokenizer,detect_model,generation_config,prompt,args.use_p,args.hard_vq,args.drop_zero,args.repetition_penalty,args.verbose)
    
    if args.type=="choice":
        parquet_path=os.path.join(args.data,"choice/author/author.parquet")
        save_json_path=os.path.join(save_dir,"author.json")
        test_choice(parquet_path,save_json_path,model,tokenizer,detect_model,generation_config,args.use_p,args.hard_vq,args.drop_zero,args.repetition_penalty,args.verbose)

        parquet_path=os.path.join(args.data,"choice/style/style.parquet")
        save_json_path=os.path.join(save_dir,"style.json")
        test_choice(parquet_path,save_json_path,model,tokenizer,detect_model,generation_config,args.use_p,args.hard_vq,args.drop_zero,args.repetition_penalty,args.verbose)

        parquet_path=os.path.join(args.data,"choice/layout/layout.parquet")
        save_json_path=os.path.join(save_dir,"layout.json")
        test_choice(parquet_path,save_json_path,model,tokenizer,detect_model,generation_config,args.use_p,args.hard_vq,args.drop_zero,args.repetition_penalty,args.verbose)

    if args.type=="bilingual":
        parquet_path=os.path.join(args.data,"reasoning/bilingual/medium/bilingual_medium.parquet")
        save_json_path=os.path.join(save_dir,"bilingual.json")
        test_bilingual(parquet_path,save_json_path,model,tokenizer,detect_model,generation_config,args.use_p,args.hard_vq,args.drop_zero,args.repetition_penalty,args.verbose)

    
    if args.type=="intent":
        parquet_path=os.path.join(args.data,"reasoning/intent/intent.parquet")
        save_json_path=os.path.join(save_dir,"intent.json")
        test_intent(parquet_path,save_json_path,model,tokenizer,detect_model,generation_config,args.use_p,args.hard_vq,args.drop_zero,args.repetition_penalty,args.verbose)



if __name__=='__main__':
    main()