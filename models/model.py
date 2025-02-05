import torch
import torch.nn as nn
import sys
import os
import json

from collections import OrderedDict
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from transformers import AutoConfig
from InternVL.modeling_intern_vit import InternVisionModel
from .perceiver_resampler import PerceiverResampler, MLP
from config.configu import device, VIT_MODEL_PATH, MLP1_PATH, TOK_EMBEDDING_PATH, TOKENIZER_PATH,NORM_TOK_EMBEDDING_PATH,NORM_PARAMS_PATH


def load_json(pth):
    """加载json文件"""
    with open(pth, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data
def load_vision_model(location='cpu'):
    vit_config = AutoConfig.from_pretrained(TOKENIZER_PATH, trust_remote_code=True).vision_config
    vision_model = InternVisionModel(vit_config).to(device).to(torch.bfloat16)
    state_dict = torch.load(VIT_MODEL_PATH, weights_only=True, map_location=location)
    incompatible_keys = vision_model.load_state_dict(state_dict)
    if incompatible_keys.unexpected_keys:
        print(f"Unexpected keys: {incompatible_keys.unexpected_keys}")
    if incompatible_keys.missing_keys:
        print(f"Missing keys: {incompatible_keys.missing_keys}")
    print("vision model已加载")
    return vision_model

def load_mlp1(downsample_ratio, vit_hidden_size=1024, llm_hidden_size=4096,location='cpu'):
    mlp1 = nn.Sequential(
        nn.LayerNorm(vit_hidden_size * int(1 / downsample_ratio) ** 2),
        nn.Linear(vit_hidden_size * int(1 / downsample_ratio) ** 2, llm_hidden_size),
        nn.GELU(),
        nn.Linear(llm_hidden_size, llm_hidden_size)
    ).to(device).to(torch.bfloat16)
    mlp1.load_state_dict(torch.load(MLP1_PATH, weights_only=True, map_location=location))
    print("mlp1已加载")
    return mlp1

def load_tok_embeddings(path=TOK_EMBEDDING_PATH,vocab_size=92553, llm_hidden_size=4096,location='cpu'):
    tok_embeddings = nn.Embedding(vocab_size, llm_hidden_size, padding_idx=2).to(device).to(torch.bfloat16)
    tok_embeddings.load_state_dict(torch.load(path, weights_only=True, map_location=location))
    print("tok_embedding已加载")
    return tok_embeddings



def load_normed_tok_embeddings(vocab_size=92553, llm_hidden_size=4096,load_checkboard=False,location="cpu"):
    tok_embeddings = nn.Embedding(vocab_size, llm_hidden_size, padding_idx=2).to(device).to(torch.bfloat16)
    tok_embeddings.load_state_dict(torch.load(NORM_TOK_EMBEDDING_PATH, weights_only=True, map_location=location))
    print("norm tok_embedding已加载")
    if load_checkboard:
        checkboard_norm=torch.load(NORM_PARAMS_PATH) # (voc_size, 2) mu sigma    pred * sigma + mu (逐行)
        print("归一化参数(mu, sigma)已加载")
        return tok_embeddings,checkboard_norm
    return tok_embeddings


def load_tokenizer():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
    return tokenizer

def load_perceiver_resampler(path=None, num_layers=4, checkpoint=None):
    model = PerceiverResampler(dim=4096, depth = num_layers).to(device).to(torch.bfloat16)
    if checkpoint == None and path!=None:
        checkpoint = torch.load(path)
    if path is not None:
        print(f"Load from {path}")
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint.keys():
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                raise FileNotFoundError("no key model_state_dict in ckpt")
        else:
            model.load_state_dict(checkpoint)
    print(f"Model has a parameter scale of {sum(p.numel() for p in model.parameters())/1e9:.3f} B.")
    return model

def load_mlp(path=None):
    model = MLP(dim=256).to(device).to(torch.bfloat16)
    if path is not None:
        model.load_state_dict(torch.load(path))
    print(f"Model has a parameter scale of {sum(p.numel() for p in model.parameters())/1e9:.3f} B.")
    return model

def load_perceiver_resampler_2(model_path, num_layers=4,device=None):
    if device==None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 初始化模型
    model = PerceiverResampler(dim=4096,depth=num_layers)
    
    # 加载预训练权重
    state_dict = torch.load(model_path, map_location='cpu')
    
    # 移除 state_dict 中的 `module.` 前缀
    state_dict = torch.load(model_path,weights_only=False)
    if 'model_state_dict' in state_dict.keys():
        state_dict = state_dict['model_state_dict']

# 3. 处理 DDP 模型的情况（检查是否有 'module.' 前缀）
    new_state_dict = OrderedDict()

    for key, value in state_dict.items():
        # 如果有 'module.' 前缀，则去掉它
        if key.startswith('module.'):
            new_key = key[len('module.'):]
        else:
            new_key = key
        new_state_dict[new_key] = value
    model = model.to_empty(device=device)
    model.load_state_dict(new_state_dict)
    
    # 将模型移动到目标设备
    
    
    # 将模型转换为所需的数据类型
    model = model.to(torch.bfloat16)
    return model

def load_pretrained_resampler(checkpoint_path, num_layers=6):
    model = load_perceiver_resampler(num_layers=num_layers)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    #print(checkpoint.keys())
    # 如果模型是通过 DDP 保存的，需要处理 'module.' 前缀
    if 'module.' in list(checkpoint.keys())[0]:
        print("load ddp Perceiver Resampler....")
        #model = torch.nn.parallel.DistributedDataParallel(model)
        model.load_state_dict(checkpoint)
    elif 'module.' in list(checkpoint['model'].keys())[0]:
        print("load ddp Perceiver Resampler....")
        #model = torch.nn.parallel.DistributedDataParallel(model)
        model.load_state_dict(checkpoint['model'])
    else:
        print("load Perseiver Resampler ...")
        model.load_state_dict(checkpoint)
    return model

def load_optimizer(optimizer, path, resume):
    # 加载checkpoint
    
    if resume:
        #assert isinstance(ckpt, dict) and 'optimizer_state_dict' in ckpt
        ckpt = torch.load(path)
        if 'optimizer_state_dict' not in ckpt:
            return optimizer
        # 处理 DDP 模型的情况
        optimizer_state_dict = ckpt['optimizer_state_dict']
        new_optimizer_state_dict = OrderedDict()

        for key, value in optimizer_state_dict.items():
            if key.startswith('module.'):
                new_key = key[len('module.'):]
            else:
                new_key = key
            new_optimizer_state_dict[new_key] = value

        optimizer.load_state_dict(new_optimizer_state_dict)
    
    return optimizer

def load_scheduler(scheduler, path, resume):
    
    if resume:
        #assert isinstance(ckpt, dict) and 'scheduler_state_dict' in ckpt
        # 加载checkpoint
        ckpt = torch.load(path)
        if 'scheduler_state_dict' not in ckpt:
            return scheduler
        # 处理 DDP 模型的情况
        scheduler_state_dict = ckpt['scheduler_state_dict']
        new_scheduler_state_dict = OrderedDict()
        for key, value in scheduler_state_dict.items():
            if key.startswith('module.'):
                new_key = key[len('module.'):]
            else:
                new_key = key
            new_scheduler_state_dict[new_key] = value

        scheduler.load_state_dict(new_scheduler_state_dict)
    
    return scheduler

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
class BoundingBoxDataset(Dataset):
    """数据集class"""
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.targets[idx]
        return x, y

class Transformer(nn.Module):
    """核心的Transformer model,encoder only"""
    def __init__(self, input_dim:int, model_dim:int, num_heads:int, num_layers:int,output_dim:int,norms=True):
        super(Transformer, self).__init__()
        self.embedding=nn.Linear(input_dim,model_dim)
        if norms:
            self.layer_norm = nn.LayerNorm(model_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads,batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers,norm=self.layer_norm if norms==True else None)
    
        self.decoder=nn.Linear(model_dim,output_dim)

    def forward(self, x):
        x=self.embedding(x)
        x = self.transformer_encoder(x)
        x=self.decoder(x)
        return x
    
class OrderFormer:
    """封装后的模型,实现数据加载,训练,测试,推理功能"""
    def __init__(self, model_path=None,max_nums=300,input_dim=4, model_dim=256, num_heads=8, num_layers=4, output_dim=1,device=torch.device("cuda"),label_name="turn",norm=False):
        self.model = Transformer(input_dim, model_dim, num_heads, num_layers, output_dim,norms=norm).to_empty(device=device)
        if isinstance(model_path,str):
            self.model.load_state_dict(torch.load(model_path))

        self.device=device
        self.max_nums=max_nums
        self.input_dim=input_dim
        self.label_name=label_name

    def _get_all_jsons(self,folder_path):
        """得到文件夹中的所有json文件路径"""
        files = os.listdir(folder_path)
        json_files = [folder_path+f for f in files if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('json')]
        return json_files
    
    def _preprocess(self,datas):
        """
        data: SHOULD BE Consistent with labelme data format
        return: 
            [
                [
                    [x1,y1,x2,y2],label
                ]
                ...
            ]
        x,y:[0,1]    
        """
        data=datas['shapes']
        h=datas['imageHeight']
        w=datas['imageWidth']
        example=[]
        X=[]
        Y=[]
        L=[]
        for obj in data:
                #记录顺序,横纵坐标
            l=obj[self.label_name]
            p=obj['points']
            X.extend([p[0][0]/w,p[1][0]/w])
            Y.extend([p[0][1]/h,p[1][1]/h])
            L.append(l)
        xmin=min(X)
        ymin=min(Y)
            #横纵坐标均减去最小值,保持平移不变性
        X=np.array(X)-xmin
        Y=np.array(Y)-ymin
        for i in range(len(L)):
            coord=[X[2*i],Y[2*i],X[2*i+1],Y[2*i+1]]
            example.append([coord,L[i]])
        return example
    def _sort_boxes(self,boxes):
        """以到(0,0)距离排序box,确保输入box是唯一的排列序列
        boxes=[[[x1,y1,x2,y2],label],...]
        label可以是标签,也可以是原始的bbox便于得到bbox和顺序的对应关系
        """
        return sorted(boxes,key=lambda x:((x[0][0]+x[0][2])/2)**2+((x[0][1]+x[0][3])/2)**2)

    def _load_data(self,path,device=torch.device("cuda"),name='turn'):
        """
        从json转为tensor的构造函数
        Args:
        path:jsons-jpgs所存在的文件夹
        max_nums:单个样本中char的最大个数
        name:取得char顺序指标的key
        Return:

        """
        max_nums=self.max_nums
        device=self.device
        all_jsons=self._get_all_jsons(path)
        raw=[]
        for j in all_jsons:
            datas=load_json(j)
            example=self._preprocess(datas)
            raw.append(example)
        transformed_inputs=[]
        transformed_labels=[]
        originNs=[]#记录原序列的长度,用于从结果中得到序列
        for item in raw:
            item=self._sort_boxes(item)
            originNs.append(len(item))
            lst=[]
            ls=[]
            for x in item:
                #lst=lst+[x1,y1,x2,y2]
                lst.extend(x[0])
                #ls记录label
                ls.append(int(x[1]))
            #pad全0序列和全0标签到指定的max_nums长度
            lst.extend([0]*self.input_dim*(max_nums-len(item)))
            ls.extend([0]*(max_nums-len(item)))
            
            transformed_inputs.append(lst)
            transformed_labels.append(ls)
        return torch.tensor(transformed_inputs,dtype=torch.float32).reshape((-1,max_nums,self.input_dim)).to(device),torch.tensor(transformed_labels,dtype=torch.float32).reshape((-1,self.max_nums,1)).to(device),originNs
    
    def _decode(self,output,N,batch_size=1):
        """从输出的tensor解码得到排序"""
        new_output=output.reshape((batch_size,-1))[:,:N]
        sorted_indices = torch.argsort(new_output, dim=1)
        ranks = torch.argsort(sorted_indices, dim=1)
        return ranks + 1
    
    def _get_acc(self,tensor1, tensor2):
        """计算两个相同形状tensor数值相同的位置的占比"""
        # Ensure the tensors are of the same shape
        assert tensor1.shape == tensor2.shape, "Tensors must have the same shape"
        
        # Create a boolean mask where the values are equal
        equal_mask = tensor1 == tensor2
        
        # Calculate the proportion of equal values
        equal_count = torch.sum(equal_mask).item()
        total_elements = torch.numel(tensor1)
        
        proportion_equal = equal_count / total_elements
        
        return proportion_equal


    def train(self, path,batch_size=4,lr=0.0002,weight_decay=0,epochs=1000,verbose=True):
        """训练函数"""
        if verbose:
            print("Loading dataset...")
        data,labels,_=self._load_data(path=path,device=self.device,name=self.label_name)

        #TODO :可指定的训练策略
        optimizer = optim.AdamW(self.model.parameters(), lr=lr,weight_decay=weight_decay,amsgrad=True)
        # scheduler=optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=10)
        scheduler=optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
        criterion=torch.nn.MSELoss()

        dataset = BoundingBoxDataset(data, labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        min_loss=float("inf")
        if verbose:
            print("Start training...")
        for epoch in range(epochs):
            losses=0
            for batch_idx,(inputs, y) in enumerate(tqdm((dataloader))):                     
                optimizer.zero_grad()
                outputs = self.model(inputs)  

                loss = criterion(outputs, y)
                loss.backward()
                losses+=loss.item()
                scheduler.step(epoch + batch_idx / len(dataloader))
                optimizer.step()
            #scheduler.step()
            
            if verbose:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {losses/len(dataloader)}")
            if losses/len(dataloader)<min_loss:
                min_loss=losses/len(dataloader)
                if verbose:
                    print("Saving best model...")
                torch.save(self.model.state_dict(),'best.pth')

 
    def eval(self, path,verbose=False):
        """在数据集上测试,计算平均loss和mAP"""
        testdata,testlabels,Ns=self._load_data(path=path,device=self.device,name=self.label_name)
        dataset = BoundingBoxDataset(testdata, testlabels)
        testloader=DataLoader(dataset,batch_size=1,shuffle=False)

        self.model.eval()
        losses=0
        mAP=0
        if verbose:
            print("Evaluation...")
        criterion = nn.MSELoss()
        for i,(inputs, y) in enumerate(testloader):
            outputs = self.model(inputs)
            pred= self._decode(outputs,Ns[i])
            gt=y.reshape((1,-1))[:,:Ns[i]]
            loss = criterion(pred, gt)
            acc=self._get_acc(pred,gt)
            if verbose:

                print("Pred:",pred)
                print("GT:",gt)
                print("loss= ",loss.item())
                print("acc= ",acc,'\n')
            losses+=loss.item()
            #mAP+=1 if acc==1 else 0
            mAP+=acc
        print(f"Test MSELoss= {losses/len(testloader):.4f}\nTest mAP= {mAP/len(testloader):.4f}")
    
    def predict(self,datas,jpg_path=None,save_path=None,verbose=False):
        """
        进行单个数据的预测,如果有图片,保存路径,可以进行verbose可视化
        返回一个dict,key是顺序,value是box的位置
        """
        if save_path:
            os.makedirs(save_path,exist_ok=True)
        import time
        st=time.time()
        data=datas['shapes']
        h=datas['imageHeight']
        w=datas['imageWidth']
        example=[]
        X=[]
        Y=[]
        Ls=[]
        for obj in data:
                #记录顺序,横纵坐标
            p=obj['points']
            flat_p=[p[0][0],p[0][1],p[1][0],p[1][1]]
            Ls.append(flat_p)
            X.extend([p[0][0]/w,p[1][0]/w])
            Y.extend([p[0][1]/h,p[1][1]/h])
        xmin=min(X)
        ymin=min(Y)
            #横纵坐标均减去最小值,保持平移不变性
        X=np.array(X)-xmin
        Y=np.array(Y)-ymin
        for i in range(len(data)):
            coord=[X[2*i],Y[2*i],X[2*i+1],Y[2*i+1]]
            example.append([coord,Ls[i]])
        example=self._sort_boxes(example)
        inputs=[]
        labels=[]
        for coord in example:
            inputs.extend(coord[0])
            labels.append(coord[1])
        inputs.extend([0]*self.input_dim*(self.max_nums-len(example)))
        
        x=torch.tensor(inputs,dtype=torch.bfloat16).reshape((-1,self.max_nums,self.input_dim)).to(self.device)
        
        mstart=time.time()
        self.model.eval()
        y=self.model(x)
        mtime=time.time()-mstart
        pred=self._decode(y,len(example)).squeeze().tolist()
        results={}
        if isinstance(pred,int):
            pred=[pred]
        for p,l in zip(pred,labels):
            results[p]=l
        
        post_start=time.time()
        results=self.postprocess(dict(sorted(results.items(), key=lambda item: item[0])),w,h,save_path,jpg_path)
        ptime=time.time()-post_start
        if verbose:
            print(f"Using {time.time()-st:.3f}s to sort boxes,with {mtime:.3f}s on OrderFormer inference,{ptime:.3f}s on postprocess.")
        if verbose and isinstance(jpg_path,str) and isinstance(save_path,str):
            import cv2
            frame = cv2.imread(jpg_path)

            for idx ,points in results.items():
                x1, y1, x2, y2 = int(points[0]), int(points[1]), int(points[2]), int(points[3])
                cv2.rectangle(frame, (x1, y1), (x2, y2), thickness=2,color=(255,0,0),lineType=cv2.LINE_AA)
                label_position = ((x1+x2)//2,(y1+y2)//2)  # Adjust the position of the label as needed
                cv2.putText(frame, str(idx), label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
            name=jpg_path.split("/")[-1]
            cv2.imwrite(save_path+"ordered_"+name,frame)
    
        return dict(sorted(results.items(), key=lambda item: item[0]))

    
    
    def postprocess(self,results,width,height,save_dir,jpg_path,vis=True,max_iters=5):
        def ordered_permute(b1,b2,b3):
            ws=[b1[2]-b1[0],b2[2]-b2[0],b3[2]-b3[0]]
            hs=[b1[3]-b1[1],b2[3]-b2[1],b3[3]-b3[1]]
            c1=[(b1[0]+b1[2])/2,(b1[1]+b1[3])/2]
            c2=[(b2[0]+b2[2])/2,(b2[1]+b2[3])/2]
            c3=[(b3[0]+b3[2])/2,(b3[1]+b3[3])/2]
            s=[ws[0]*hs[0],ws[1]*hs[1],ws[2]*hs[2]]
            if max(abs(c1[1]-c2[1]),abs(c1[1]-c3[1]),abs(c2[1]-c3[1]))<min(hs) and min(s)/max(s)>0.7:
                c=[c1[0],c2[0],c3[0]]

            else:
                c=[3,2,1]
            indexed_c = list(enumerate(c))  

            
            sorted_by_value = sorted(indexed_c, key=lambda x: x[1],reverse=True)  

      
            sorted_indices = [index for index, value in sorted_by_value]

            return sorted_indices
        index=list(results.keys())
        boxes=[[item[0]/width,item[1]/height,item[2]/width,item[3]/height] for item in list(results.values())]
        for i in range(len(index)-2):
            now=boxes[i]
            next_1=boxes[i+1]
            next_2=boxes[i+2]
            order=ordered_permute(now,next_1,next_2)
            
            j=i+1
            boxes[i],boxes[i+1],boxes[i+2]=boxes[i+order[0]],boxes[i+order[1]],boxes[i+order[2]]
            results[j],results[j+1],results[j+2]=results[j+order[0]],results[j+order[1]],results[j+order[2]]
            
        return results
    
def load_orderformer(path,
    max_num=50,
    input_dim=4,
    output_dim=1,
    model_dim=256,
    num_layers=4,
    num_heads=8,
    ):

    model=OrderFormer(max_nums=max_num,
                        num_layers=num_layers,
                        input_dim=input_dim,
                        output_dim=output_dim,
                        model_dim=model_dim,
                        num_heads=num_heads,
                        model_path=path,
                        label_name='turn',
                        norm=False)
    return model