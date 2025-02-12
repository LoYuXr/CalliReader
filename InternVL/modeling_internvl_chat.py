import cv2
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from config.configu import *
from models.model import *
from models.similarity import *
from sklearn.cluster import KMeans
from utils.utils import *
import warnings
from typing import Any, List, Optional, Tuple, Union
import torch
import random
import torch.utils.checkpoint
import transformers
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import (AutoModel, GenerationConfig, LlamaForCausalLM,
                          LlamaTokenizer)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput, logging

from .configuration_internvl_chat import InternVLChatConfig
from .conversation import get_conv_template
from .modeling_intern_vit import InternVisionModel
from .modeling_internlm2 import InternLM2ForCausalLM

logger = logging.get_logger(__name__)

def coord_transform(box,return_4=True):
    if return_4:
        return [box[0][0],box[0][1],box[1][0],box[1][1]]
    else:
        return [[box[0],box[1]],[box[2],box[3]]]
def insert_zeros(input_ids, attention_mask, num_zeros=5):

    device = input_ids.device  # 获取原始设备
    input_ids = input_ids.cpu().clone()  # 将张量移到 CPU 并克隆
    attention_mask = attention_mask.cpu().clone()  # 将张量移到 CPU 并克隆

    for _ in range(num_zeros):
        # 随机选择插入位置
        insert_pos = random.randint(0, input_ids.size(1))
        
        # 在 input_ids 中插入 0
        input_ids = torch.cat((input_ids[:, :insert_pos], torch.tensor([[0]]), input_ids[:, insert_pos:]), dim=1)
        
        # 在 attention_mask 中插入 1
        attention_mask = torch.cat((attention_mask[:, :insert_pos], torch.tensor([[1]]), attention_mask[:, insert_pos:]), dim=1)

     # 将张量移回原始设备
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    return input_ids, attention_mask


def add_Gaussian_noise(input_embeds, rate=1e-1):

    device = input_embeds.device
    input_embeds = input_embeds.cpu().clone()

    mean = input_embeds.mean()
    std = input_embeds.std()
    noise = torch.randn(input_embeds.size()) * std + mean
    noisy_input_embeds = input_embeds + rate * noise

    noisy_input_embeds = noisy_input_embeds.to(device)
    noisy_input_embeds = noisy_input_embeds.to(torch.bfloat16)

    return noisy_input_embeds


def version_cmp(v1, v2, op='eq'):
    import operator

    from packaging import version
    op_func = getattr(operator, op)
    return op_func(version.parse(v1), version.parse(v2))

def most_frequent_rgb(image_array):
    """找一张图片中最frequent的rgb,用于填充mask"""
    # Flatten the image array to a 2D array where each row is an RGB tuple
    pixels = image_array.reshape(-1, image_array.shape[-1])
    
    # Use np.unique with return_counts to find unique rows and their counts
    unique_pixels, counts = np.unique(pixels, axis=0, return_counts=True)
    
    # Find the index of the most frequent pixel
    most_frequent_index = np.argmax(counts)
    
    # Get the most frequent pixel and its count
    most_frequent_pixel = unique_pixels[most_frequent_index]
    frequency = counts[most_frequent_index]
    return most_frequent_pixel, frequency

def most_frequent_rgb_fast(image_array):
    """快速查找图片中最频繁的RGB值，不返回频率"""
    # 将RGB每个通道的值映射为一个唯一的整数，形如 R * 256^2 + G * 256 + B
    flattened = image_array.reshape(-1, 3)
    rgb_ints = flattened[:, 0] * 256**2 + flattened[:, 1] * 256 + flattened[:, 2]

    # 使用np.bincount统计每个唯一RGB组合出现的次数
    counts = np.bincount(rgb_ints)

    # 找到出现次数最多的那个整数
    most_frequent_index = np.argmax(counts)

    # 将整数转换回RGB值
    r = (most_frequent_index // 256**2) % 256
    g = (most_frequent_index // 256) % 256
    b = most_frequent_index % 256

    return (r, g, b)



def mask_area(image_array,coords,color):
    """对一张图片在框定的一系列box进行mask"""
    # Define the bounding box (x1, y1, x2, y2)
    #color=average_rgb(modified_image)
    for coord in coords:
        x1, y1, x2, y2 = coord 
        image_array[y1:y2, x1:x2] =color  # 255 for white in an RGB image

    return image_array


class InternVLChatModel(PreTrainedModel):
    config_class = InternVLChatConfig
    main_input_name = 'pixel_values'
    _supports_flash_attn_2 = True
    _no_split_modules = ['InternVisionModel', 'LlamaDecoderLayer', 'InternLM2DecoderLayer']

    def __init__(self, config: InternVLChatConfig, vision_model=None, language_model=None):
        super().__init__(config)

        assert version_cmp(transformers.__version__, '4.36.2', 'ge')
        image_size = config.force_image_size or config.vision_config.image_size
        patch_size = config.vision_config.patch_size
        self.patch_size = patch_size
        self.select_layer = config.select_layer
        self.template = config.template
        ##TODO change the number of img tokens
        self.num_image_token = int((image_size // patch_size) ** 2 * (config.downsample_ratio ** 2))
        #self.num_image_token = 3
        self.downsample_ratio = config.downsample_ratio
        self.ps_version = config.ps_version


        
        self.mu_sigma=torch.load(NORM_PARAMS_PATH)['weight']
        self.mu=self.mu_sigma[:,0].reshape((-1,1))
        self.sigma=self.mu_sigma[:,1].reshape((-1,1)) #[vocab_size, 1]
        self.normed_emb,self.mu_sigma=self.load_normed_tok_embeddings(load_checkboard=True)
        self.resampler=load_perceiver_resampler_2(PERCEIVER_CHECKPOINT,num_layers=4)
    
        self.sorter=load_orderformer(ORDERFORMER_CHECKPOINT)


        logger.info(f'num_image_token: {self.num_image_token}')
        logger.info(f'ps_version: {self.ps_version}')
        # print('vision_model', vision_model)
        # print('language_model', language_model)
        # print('config.llm_config.architectures[0]', config.llm_config.architectures[0])
        if vision_model is not None:
            self.vision_model = vision_model
        else:
            self.vision_model = InternVisionModel(config.vision_config)
        if language_model is not None:
            self.language_model = language_model
        else:
            if config.llm_config.architectures[0] == 'LlamaForCausalLM':
                self.language_model = LlamaForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == 'InternLM2ForCausalLM':
                self.language_model = InternLM2ForCausalLM(config.llm_config)
            else:
                raise NotImplementedError(f'{config.llm_config.architectures[0]} is not implemented.')
  

        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.llm_config.hidden_size

        self.mlp1 = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),
            nn.Linear(vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size)
        )
        
        self.img_context_token_id = None
        self.conv_template = get_conv_template(self.template)
        self.system_message = self.conv_template.system_message
    def load_normed_tok_embeddings(self,vocab_size=92553, llm_hidden_size=4096,load_checkboard=False):
        tok_embeddings = nn.Embedding(vocab_size, llm_hidden_size, padding_idx=2).to_empty(device=torch.device('cuda')).to(torch.bfloat16)
        tok_embeddings.load_state_dict(torch.load(NORM_TOK_EMBEDDING_PATH, weights_only=True, map_location="cpu"))
        if load_checkboard:
            checkboard_norm=torch.load(NORM_PARAMS_PATH) # (voc_size, 2) mu sigma    pred * sigma + mu (逐行)
    
            return tok_embeddings,checkboard_norm['weight']
        return tok_embeddings
    
    def forward(
            self,
            pixel_values: torch.FloatTensor,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            image_flags: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        image_flags = image_flags.squeeze(-1)
        input_embeds = self.language_model.get_input_embeddings()(input_ids)

        vit_embeds = self.extract_feature(pixel_values)
        vit_embeds = vit_embeds[image_flags == 1]
        vit_batch_size = pixel_values.shape[0]

        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        if torch.distributed.get_rank() == 0:
            print(f'dynamic ViT batch size: {vit_batch_size}, images per sample: {vit_batch_size / B}, dynamic token length: {N}')

        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == self.img_context_token_id)
        try:
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
        except Exception as e:
            vit_embeds = vit_embeds.reshape(-1, C)
            print(f'warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, '
                  f'vit_embeds.shape={vit_embeds.shape}')
            n_token = selected.sum()
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds[:n_token]

        input_embeds = input_embeds.reshape(B, N, C)

        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                   int(c / (scale_factor * scale_factor)))
        if self.ps_version == 'v1':
            warnings.warn("In ps_version 'v1', the height and width have not been swapped back, "
                          'which results in a transposed image.')
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def extract_feature(self, pixel_values):
        if self.select_layer == -1:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=False,
                return_dict=True).last_hidden_state
        else:

            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True).hidden_states[self.select_layer]
        vit_embeds = vit_embeds[:, 1:, :]

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])

        vit_embeds = self.mlp1(vit_embeds)
        return vit_embeds
    
    @torch.no_grad()
    def calli_align(self,img_path,detect_model, drop_zero = False, use_hard_vector_quant=False,save_path=None,verbose=False):
        def dynamic_read(img_path,mode='c'):
    # 如果是字符串类型（文件路径），用 cv2 读取
            if isinstance(img_path, str):
                img = cv2.imread(img_path)
                
                if img is None:
                    try:
                        img = Image.open(img_path).convert("RGB")
                        img = np.array(img)
                    except:
                        raise ValueError(f"Image at path {img_path} could not be loaded.")            
            # 如果是 PIL.Image.Image 类型，将其转为 cv2 格式
            elif isinstance(img_path, Image.Image):
                img = np.array(img_path)  # PIL 转 numpy 数组
                # 因为 OpenCV 是 BGR，需要将 RGB 转为 BGR
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            else:
                raise TypeError(f"Unsupported image type: {type(img_path)}")
            if mode=='i':
                img=Image.fromarray(img).convert("RGB")
            return img
        import time
        def iterative_only_boxes(model,jpg_path):
    
            image = dynamic_read(jpg_path)
            
            image_array = np.array(image)
       
            h, w, channels = image.shape
            boxes=[]
            
           
            color=most_frequent_rgb_fast(image_array)
            while True:
                res=model(image_array,verbose=False)[0]
              
                to_be_masked=[]
                for box in res.boxes:
                    xyxy = box.xyxy.squeeze().tolist()
                    x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                    to_be_masked.append([x1,y1,x2,y2])
                boxes.extend(to_be_masked)
                if len(to_be_masked)>250:
                    image_array=mask_area(image_array,to_be_masked,color)
                else:
                    break
                
            boxes=[[[max(item[0],0),max(item[1],0)],[min(item[2],w),min(item[3],h)]]for item in boxes]
       

            i=0
            length=len(boxes)
            while i<length:
                j=0
                main_box=boxes[i]
                while j<length:
                    if i==j:
                        j+=1
                        continue
                    iou=calculate_iou(coord_transform(main_box),coord_transform(boxes[j])) 
                    if iou>0.8:
                        rm = boxes[j]
                        boxes.remove(rm)
                        if j<i:
                            i-=1
                        length-=1
                        j-=1
                    j+=1
                i+=1

            return boxes
        def char2col_with_kmeans(jpg_path,boxes, verbose=False):
            ## modified
            def kmeans_boxes(bounding_boxes):
                areas = [  (box[1][0] - box[0][0])*(box[1][1] - box[0][1]) for box in bounding_boxes]


                # 转换为 numpy 数组
                areas = np.array(areas).reshape(-1, 1)

                # 使用 KMeans 进行聚类，将面积分为两组
                kmeans = KMeans(n_clusters=2, random_state=0).fit(areas)

                # 获取每个 bounding box 的标签
                labels = kmeans.labels_

                # 根据标签将 bounding boxes 分成两个组
                group_0 = []
                group_1 = []

                for i, label in enumerate(labels):
                    if label == 0:
                        group_0.append(bounding_boxes[i])
                    else:
                        group_1.append(bounding_boxes[i])
                
                group_0 = sorted(group_0, key = lambda x: (x[1][0]-x[0][0]), reverse=True)
                group_1 = sorted(group_1, key = lambda x: (x[1][0]-x[0][0]), reverse=True)

                if (group_1[0][1][0] - group_1[0][0][0]) > (group_0[0][1][0] - group_0[0][0][0]):# and len(group_1) > 0.8*len(group_0): # 1 为正文，0为落款
                    g1_hs = np.array([x[1][1]-x[0][1] for x in group_1]).mean()
                    thr1 = 1*( group_1[-1][1][0] - group_1[-1][0][0])
                    thr2 = 0.8*g1_hs
                    #luokuan_mean_area = np.array([(ele[1][0] - ele[0][0])*(ele[1][1] - ele[0][1]) for ele in group_0]).mean()
                    new_0 = []
                    for ele in group_0:
                        if (ele[1][0] - ele[0][0]) >= thr1 or (ele[1][1] - ele[0][1]) >= thr2 or (areas.min()/(ele[1][0] - ele[0][0])*(ele[1][1] - ele[0][1]) <= 1/5 and areas.mean() / ((ele[1][0] - ele[0][0])*(ele[1][1] - ele[0][1])) <= 1.3):
                            group_1.append(ele)
                        else:
                            new_0.append(ele)

                    grouped_luokuan = merge_boxes(new_0.copy())
                   
                    final_ = []
                    for ele in new_0:
                        if ele in grouped_luokuan:
                        
                            group_1.append(ele)
                        else:
                            final_.append(ele)
                    group_0 = final_
                
                elif (group_0[0][1][0] - group_0[0][0][0]) > (group_1[0][1][0] - group_1[0][0][0]):# and len(group_0) > 0.8*len(group_1):
                    g0_hs = np.array([x[1][1]-x[0][1] for x in group_0]).mean()
                    thr1 = 1*( group_0[-1][1][0] - group_0[-1][0][0])
                    thr2 = 0.8*g0_hs
                    #luokuan_mean_area = np.array([(ele[1][0] - ele[0][0])*(ele[1][1] - ele[0][1]) for ele in group_1]).mean()
                    new_1 = []
                    for ele in group_1:
                        if (ele[1][0] - ele[0][0]) >= thr1 or (ele[1][1] - ele[0][1]) >= thr2 or (areas.min()/(ele[1][0] - ele[0][0])*(ele[1][1] - ele[0][1]) <= 1/5 and areas.mean() / ((ele[1][0] - ele[0][0])*(ele[1][1] - ele[0][1])) <=1.3):
                          
                            group_0.append(ele)
                        else:
                            new_1.append(ele)
                    
                    grouped_luokuan = merge_boxes(new_1.copy())
                    
                    final_ = []
                    for ele in new_1:
                        if ele in grouped_luokuan:
                            group_0.append(ele)
                        else:
                            final_.append(ele)
                    group_1 = final_
  
                return group_0,group_1

            def toint(lst):
                if len(lst)==2:
                    return [[int(lst[0][0]),int(lst[0][1])],[int(lst[1][0]),int(lst[1][1])]]
                else:
                    return [int(lst[0]),int(lst[1]),int(lst[2]),int(lst[3])]
            img = dynamic_read(jpg_path)
            h, w, channels = img.shape

            normalized_boxes=[[[item[0][0]/w,item[0][1]/h],[item[1][0]/w,item[1][1]/h]] for item in boxes]
            S=np.array([(item[0][0]-item[1][0])*(item[0][1]-item[1][1]) for item in normalized_boxes])
            # print(np.max(S)-np.min(S),h,w)
            # print(boxes)
            # print(normalized_boxes)
        
            coef_var=np.std(S)/np.mean(S)
            boxes2class=None
            col2class=None
            
            if coef_var>0.66 and S.min()/S.mean() <= 1/8:
                
                boxes1,boxes2=kmeans_boxes(normalized_boxes)
            
                
                boxes1=[[[item[0][0]*w,item[0][1]*h],[item[1][0]*w,item[1][1]*h]] for item in boxes1]
                boxes2=[[[item[0][0]*w,item[0][1]*h],[item[1][0]*w,item[1][1]*h]] for item in boxes2]
                columns1=merge_boxes(boxes1.copy())
                columns2=merge_boxes(boxes2.copy())
                
                columns=columns1+columns2
                boxes2class={1:[toint(item) for item in boxes1],2:[toint(item) for item in boxes2]}
                col2class={1:[toint(item) for item in columns1],2:[toint(item) for item in columns2]}
                #[[481.3252033886607, 1185.3073037637248], [748.9909909909909, 1616.216216216216]]

            else:
                columns=merge_boxes(boxes.copy())


            results={"imageHeight":h,"imageWidth":w,"shapes":[{"points":toint(col)} for col in columns],
                    "boxes2class":boxes2class,"col2class":col2class}
            
           
            #print("saving results...")

            # if verbose:
            #     frame = dynamic_read(jpg_path)
            #     name=jpg_path.split("/")[-1]
            #     os.makedirs("./detect_boxes_char2col/result_merge", exist_ok=True)
            #     for i,box in enumerate(results['shapes']):
                
            #             xyxy = box['points']
            #             x1, y1, x2, y2 = int(xyxy[0][0]), int(xyxy[0][1]), int(xyxy[1][0]), int(xyxy[1][1])
            #             colo = (255,0,0)
            #             cv2.rectangle(frame, (x1, y1), (x2, y2), thickness=2,color=colo,lineType=cv2.LINE_AA)
            #             # put labels
                        
            #             if boxes2class is not None:
            #                 if xyxy in col2class[1]:
            #                     cv2.putText(frame, str(1), ((x1+x2)//2, (y1+y2)//2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, colo, thickness=2, lineType=cv2.LINE_AA)
            #                 elif xyxy in col2class[2]:
            #                     cv2.putText(frame, str(2), ((x1+x2)//2, (y1+y2)//2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 30, 235), thickness=2, lineType=cv2.LINE_AA)
            #             #cv2.putText(frame, str(i+1), ((x1+x2)//2, (y1+y2)//2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, colo, thickness=2, lineType=cv2.LINE_AA)
            #     cv2.imwrite("./detect_boxes_char2col/result_merge"+name,frame)
            return results
        
        def sort_boxes(jpg,detector,model,thres=0.8):
        
            boxes=iterative_only_boxes(detector,jpg)
            
            data=char2col_with_kmeans(jpg,boxes,verbose=False)
            
            res=model.predict(data,jpg)
            final_results=[]
            for idx,col in res.items(): 
                lst=[]
                for item in boxes:
                    ratio=calculate_iou(col,[item[0][0],item[0][1],item[1][0],item[1][1]],mini=True)
                   
                    if ratio>=thres:
                        lst.append([item[0][0],item[0][1],item[1][0],item[1][1]])
                lst=sorted(lst, key=lambda item: (item[1]+item[3])/2)
                final_results.extend(lst)
            #print(len(boxes),len(res),len(final_results))
            return final_results
        if img_path is None:
            return None,None
        
        st=time.time()
        boxes=sort_boxes(img_path,detect_model,self.sorter)
        ed=time.time()
        if verbose:
            print(f"YOLO+Orderformer {ed-st:.2f}s")
        if save_path!=None:
            frame = dynamic_read(img_path)
            name=img_path.split("/")[-1]
            for i,box in enumerate(boxes):
            
                xyxy = box
                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                colo = (255,0,0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), thickness=2,color=colo,lineType=cv2.LINE_AA)
                    # put labels
                cv2.putText(frame, str(i+1), ((x1+x2)//2, (y1+y2)//2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, colo, thickness=2, lineType=cv2.LINE_AA)
            print(save_path+"oredered_result_"+name)
            cv2.imwrite(save_path+"oredered_result_"+name,frame)
        
        st=time.time()
        pixel_values=[]
        img=np.array(dynamic_read(img_path,mode='i').convert("RGB"))
        
        for xyxy in boxes:
            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
            sub_img=Image.fromarray(img[y1:y2,x1:x2])
            pixel_values.append(load_image_2(sub_img).to(torch.bfloat16).cuda())
        ed1=time.time()
        results=torch.cat(pixel_values)
            
        image_embeddings=self.extract_feature(results)
        ed2=time.time()
        output=self.resampler(image_embeddings)
        ed3=time.time()
 
        #TODO 可以indices转换回去
        
        outs=vq_cos_sim(self.normed_emb,output, use_hard_vector_quant) #(B, 3) #如果use_vq的话现在改成dynamic: 对于max cos_sim小于等于thresh的，使用向量量化进行替换
        
        ed4=time.time()
        if verbose:
            print(f"Get pixel values {ed1-st:.2f}s")
            print(f"extract feat {ed2-ed1:.2f}s")
            print(f"Resampler forward {ed3-ed2:.2f}")
            print(f"vq cos sim {ed4-ed3:.2f}s")
        if use_hard_vector_quant:
            indices, cos_sim_values = outs
            #### DEFINE THRESH!!!
            thresh = 0.5
        else:
            indices = outs

        if use_hard_vector_quant:
            print("Dynamic vector quantization...")
        
            below_mask = (cos_sim_values <= thresh).to(torch.bfloat16).unsqueeze(-1)
            
            output = output * (1-below_mask) + self.normed_emb.weight[indices] * below_mask
        
            
        flattened_output = output.view(-1, output.shape[-1])
        flattened_indices = indices.view(-1)

        if drop_zero:            
            filtered_indices=flattened_indices[flattened_indices!=0]
            filtered_output=flattened_output[flattened_indices!=0]   


            sigma_flat = self.sigma[filtered_indices]  # 形状 (183 * 3, 1)
            mu_flat = self.mu[filtered_indices]

            sigma_flat = sigma_flat.expand(-1, filtered_output.shape[-1])
            mu_flat = mu_flat.expand(-1, filtered_output.shape[-1])
            back_to_origin_flat = filtered_output * sigma_flat + mu_flat
        
        else:
            sigma_flat = self.sigma[flattened_indices]
            mu_flat = self.mu[flattened_indices]
            sigma_flat = sigma_flat.expand(-1, flattened_output.shape[-1])
            mu_flat = mu_flat.expand(-1, flattened_output.shape[-1])
            back_to_origin_flat = flattened_output * sigma_flat + mu_flat
       
        
        return back_to_origin_flat, indices

    def find_coordinates(self,text):
        import re

        numbers = re.findall(r'\d+', text)

        numbers = [int(num) for num in numbers]  # 如果需要浮点数，可以用 float()
        return numbers
    def chat_ocr(self, tokenizer, detect_model,img_path, questions, generation_config, num_patches_list=None,
                   history=None, return_history=False, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>',
                   IMG_CONTEXT_TOKEN='<IMG_CONTEXT>', ALIGNED_TOKEN="[UNUSED_TOKEN_140]",verbose=False, image_counts=None,batch=False,
                   use_p=True, drop_zero=False, hard_vq=False, repetition_penalty=1.5,region_wise=False):




        pixel_values = None
        if img_path is not None:
            try:
                if region_wise:
                    img=np.array(Image.open(img_path).convert("RGB"))
                    coord=self.find_coordinates(questions)
                    x1,x2,y1,y2=coord
                    sub_img=Image.fromarray(img[y1:y2,x1:x2])
                  
                    questions="输出图片中所有文字:"
                    pixel_values=load_image(sub_img).to(torch.bfloat16).to(torch.device("cuda"))
                else:
                    pixel_values=load_image(img_path).to(torch.bfloat16).to(torch.device("cuda"))
            except:
                raise FileNotFoundError
        if use_p:
            import time
            st=time.time()
            if region_wise:
                try:
                    out_tokens, indices =self.calli_align(sub_img,detect_model, drop_zero = drop_zero, use_hard_vector_quant=hard_vq,verbose=verbose)
                except:
                    return "检测失败"
            else:
                
                    out_tokens, indices =self.calli_align(img_path,detect_model, drop_zero = drop_zero, use_hard_vector_quant=hard_vq,verbose=verbose) #,tokenizer=tokenizer)
            if verbose:   
                print(f"Calli Align: {time.time()-st:.2f}s")
        # 删掉多余0
        # indices 备用，因为我们也想未来看仅使用calliAlign效果
        if pixel_values is None:
            question=questions

        if pixel_values is not None and '<image>' not in questions:
            question = '<image>\n' + questions
            #question = questions
        elif  history is None and pixel_values is None:
            question=questions
        elif '<image>' in questions:
            question=questions

        if history is None and use_p and '[UNUSED_TOKEN_140]' not in question:
            question =question+'[UNUSED_TOKEN_140]'*out_tokens.shape[0]
        if num_patches_list is None:
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        template = get_conv_template(self.template)
        template.system_message = self.system_message
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep)

        history = [] if history is None else history
        for (old_question, old_answer) in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        

        for num_patches in num_patches_list:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN

            query = query.replace('<image>', image_tokens, 1)
       
        model_inputs = tokenizer(query, return_tensors='pt')

        input_ids = model_inputs['input_ids'].cuda()
       
        attention_mask = model_inputs['attention_mask'].cuda()

        generation_config['eos_token_id'] = eos_token_id


        if use_p:
            generation_output = self.generate_ocr(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                reference_embeds=out_tokens,
                repetition_penalty=repetition_penalty,
                **generation_config
            )
        else:
            generation_output = self.generate_ocr(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                repetition_penalty=repetition_penalty,
                **generation_config
            )
        response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
        response = response.split(template.sep)[0].strip()
        history.append((question, response))
        if return_history:
            return response, history
        else:
            query_to_print = query.replace(IMG_CONTEXT_TOKEN, '')
            query_to_print = query_to_print.replace(f'{IMG_START_TOKEN}{IMG_END_TOKEN}', '<image>')


            return response
    
    
    def dynamic_chat(self, tokenizer, pixel_values, questions, generation_config, num_patches_list=None,
                   history=None, return_history=False, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>',
                   IMG_CONTEXT_TOKEN='<IMG_CONTEXT>', verbose=False, image_counts=None,batch=False,use_p=True):
        if use_p:
            self.num_image_token=3
        if batch:
            assert isinstance(questions,list) and len(questions)>0 and isinstance(questions[0],str)
            if history is not None or return_history:
                print('Now multi-turn chat is not supported in batch_chat.')
                raise NotImplementedError

            if image_counts is not None:
                num_patches_list = image_counts
                print('Warning: `image_counts` is deprecated. Please use `num_patches_list` instead.')

            img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
            self.img_context_token_id = img_context_token_id

            if verbose and pixel_values is not None:
                image_bs = pixel_values.shape[0]
                print(f'dynamic ViT batch size: {image_bs}')

            queries = []
            for idx, num_patches in enumerate(num_patches_list):
                question = questions[idx]
                if pixel_values is not None and '<image>' not in question:
                    question = '<image>\n' + question
                template = get_conv_template(self.template)
                template.append_message(template.roles[0], question)
                template.append_message(template.roles[1], None)
                query = template.get_prompt()

                image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
                query = query.replace('<image>', image_tokens, 1)
                queries.append(query)

            # print(query)
            tokenizer.padding_side = 'left'
            model_inputs = tokenizer(queries, return_tensors='pt', padding=True)
            input_ids = model_inputs['input_ids'].cuda()
            attention_mask = model_inputs['attention_mask'].cuda()
            eos_token_id = tokenizer.convert_tokens_to_ids(template.sep)
            generation_config['eos_token_id'] = eos_token_id
            if use_p:
                generation_output = self.generate(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **generation_config
                )
            else:
                
                generation_output = self.generate_origin(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **generation_config
                )
            responses = tokenizer.batch_decode(generation_output, skip_special_tokens=True)
            responses = [response.split(template.sep)[0].strip() for response in responses]
            return responses
        else:
            assert isinstance(questions,str)
            if num_patches_list is None:
                num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
            assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

            img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
            self.img_context_token_id = img_context_token_id

            template = get_conv_template(self.template)
            template.system_message = self.system_message
            eos_token_id = tokenizer.convert_tokens_to_ids(template.sep)

            history = [] if history is None else history
            for (old_question, old_answer) in history:
                template.append_message(template.roles[0], old_question)
                template.append_message(template.roles[1], old_answer)
            template.append_message(template.roles[0], questions)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()


            if verbose and pixel_values is not None:
                image_bs = pixel_values.shape[0]
                print(f'dynamic ViT batch size: {image_bs}')


            # print('num_image_token', self.num_image_token)
            # print('num_patches_list', num_patches_list)


            query=f"""<|im_start|>system你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。<|im_end|>\n<|im_start|>user{questions}"""
            query = query+'<image>'
            for num_patches in num_patches_list:
                #image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
                image_tokens =  IMG_CONTEXT_TOKEN * self.num_image_token 
                #print('tokens_num', len(image_tokens))
                query = query.replace('<image>', image_tokens, 1)
            
            query+="<|im_end|>\n<|im_start|>assistant"
            # print(query)
            model_inputs = tokenizer(query, return_tensors='pt')
 

            input_ids = model_inputs['input_ids'].cuda()
            attention_mask = model_inputs['attention_mask'].cuda()


            generation_config['eos_token_id'] = eos_token_id
            if use_p:
                
                generation_output = self.generate(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **generation_config
                )
            else:
                generation_output = self.generate_origin(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **generation_config
                )
            response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
            response = response.split(template.sep)[0].strip()
            history.append((questions, response))
            if return_history:
                return response, history
            else:
                query_to_print = query.replace(IMG_CONTEXT_TOKEN, '')
                query_to_print = query_to_print.replace(f'{IMG_START_TOKEN}{IMG_END_TOKEN}', '<image>')
                if verbose:
                    print(query_to_print, response)

                return response

    def batch_chat(self, tokenizer, pixel_values, questions, generation_config, num_patches_list=None,
                   history=None, return_history=False, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>',
                   IMG_CONTEXT_TOKEN='<IMG_CONTEXT>', verbose=False, image_counts=None):
        
        if history is not None or return_history:
            print('Now multi-turn chat is not supported in batch_chat.')
            raise NotImplementedError

        if image_counts is not None:
            num_patches_list = image_counts
            print('Warning: `image_counts` is deprecated. Please use `num_patches_list` instead.')

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        queries = []
        for idx, num_patches in enumerate(num_patches_list):
            question = questions[idx]
            if pixel_values is not None and '<image>' not in question:
                question = '<image>\n' + question
            template = get_conv_template(self.template)
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()

            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)
            queries.append(query)

        # print(query)
        tokenizer.padding_side = 'left'
        model_inputs = tokenizer(queries, return_tensors='pt', padding=True)
        input_ids = model_inputs['input_ids'].cuda()
        attention_mask = model_inputs['attention_mask'].cuda()
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep)
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate_origin(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        responses = tokenizer.batch_decode(generation_output, skip_special_tokens=True)
        responses = [response.split(template.sep)[0].strip() for response in responses]
        return responses


    #When call internvl,this func is called
    def chat(self, tokenizer, pixel_values, question, generation_config, history=None, return_history=False,
             num_patches_list=None, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>', IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
             verbose=False):
        #self.num_image_token=3
        # original_question = question
        if history is None and pixel_values is not None and '<image>' not in question:
            question = '<image>\n' + question

        if num_patches_list is None:
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        template = get_conv_template(self.template)
        template.system_message = self.system_message
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep)

        history = [] if history is None else history
        for (old_question, old_answer) in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()


        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]



    
        for num_patches in num_patches_list:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)
            print(num_patches,self.num_image_token)
            print(pixel_values.shape[0])
     
        model_inputs = tokenizer(query, return_tensors='pt')

        input_ids = model_inputs['input_ids'].cuda()
        attention_mask = model_inputs['attention_mask'].cuda()

        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate_origin(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
        response = response.split(template.sep)[0].strip()
        history.append((question, response))
        if return_history:
            return response, history
        else:
            query_to_print = query.replace(IMG_CONTEXT_TOKEN, '')
            query_to_print = query_to_print.replace(f'{IMG_START_TOKEN}{IMG_END_TOKEN}', '<image>')
            if verbose:
                print(query_to_print, response)

            return response
    
    @torch.no_grad()
    def generate_origin(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            input_ids: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            visual_features: Optional[torch.FloatTensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **generate_kwargs,
    ) -> torch.LongTensor:

        assert self.img_context_token_id is not None
        if pixel_values is not None:
            if visual_features is not None:
                vit_embeds = visual_features
            else:
                vit_embeds = self.extract_feature(pixel_values)
            input_embeds = self.language_model.get_input_embeddings()(input_ids)


            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)

            input_ids = input_ids.reshape(B * N)
            selected = (input_ids == self.img_context_token_id)
            assert selected.sum() != 0
            input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)
            print("ID: ",self.img_context_token_id)
            input_embeds = input_embeds.reshape(B, N, C)
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)
          

        outputs = self.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_cache=True,
            **generate_kwargs,
        )

        return outputs
    @torch.no_grad()
    def generate_ocr(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            input_ids: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            visual_features: Optional[torch.FloatTensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            reference_embeds=None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            repetition_penalty=1.5,
            **generate_kwargs,
    ) -> torch.LongTensor:

        assert self.img_context_token_id is not None
        if pixel_values is not None:
            if visual_features is not None:
                vit_embeds = visual_features
            else:
                vit_embeds = self.extract_feature(pixel_values)
            input_embeds = self.language_model.get_input_embeddings()(input_ids)
    

            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)

            input_ids = input_ids.reshape(B * N)
            selected = (input_ids == self.img_context_token_id)
            assert selected.sum() != 0
            input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)
            

            if reference_embeds is not None:
                selected = (input_ids == 92537)
                assert selected.sum() != 0
                input_embeds[selected] =reference_embeds.reshape(-1, C).to(input_embeds.device)
      

            input_embeds = input_embeds.reshape(B, N, C)
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)
         
         

        outputs = self.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_cache=True,
            repetition_penalty=repetition_penalty,
            **generate_kwargs,
        )

        return outputs
    @torch.no_grad()
    def generate(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            input_ids: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            visual_features: Optional[torch.FloatTensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **generate_kwargs,
    ) -> torch.LongTensor:

        assert self.img_context_token_id is not None
        if pixel_values is not None:
            if visual_features is not None:
                vit_embeds = visual_features
            else:
            
                vit_embeds = self.extract_feature(pixel_values)
            
            input_embeds = self.language_model.get_input_embeddings()(input_ids)
           
            vit_embeds = self.resampler(vit_embeds)
        
            
            mu=self.mu_sigma[:,0].reshape((-1,1))
            sigma=self.mu_sigma[:,1].reshape((-1,1))

            indices=vq_cos_sim(self.normed_emb,vit_embeds).reshape((-1,))
            

            vit_embeds=vit_embeds.reshape((-1,vit_embeds.shape[-1]))*sigma[indices][:]+mu[indices][:]
            
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)

            input_ids = input_ids.reshape(B * N)
            selected = (input_ids == self.img_context_token_id)
            
            assert selected.sum() != 0
          
            input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

        

            input_embeds = input_embeds.reshape(B, N, C)
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)

        outputs = self.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_cache=True,
            **generate_kwargs,
        )

        return outputs
