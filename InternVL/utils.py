import json
from PIL import Image
import numpy as np
from copy import deepcopy
import cv2
import os
from tqdm import tqdm
import shutil

def calculate_iou(boxA, boxB,mini=False):
    # 计算交集矩形的坐标
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # 计算交集面积
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # 计算两个边界框的面积
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # 计算并集面积
    unionArea = boxAArea + boxBArea - interArea

    # 计算IoU
    iou = interArea / unionArea
    if mini:
        iou=interArea/min(boxAArea,boxBArea)
    return iou
def get_all_jpgs(folder_path,suffix='.jpg'):
    """得到文件夹中的所有jpg文件路径"""
    files = os.listdir(folder_path)
    jpg_files = [folder_path+f for f in files if os.path.isfile(os.path.join(folder_path, f)) and f.endswith(suffix)]
    return jpg_files

def get_all_jsons(folder_path):
    """得到文件夹中的所有json文件路径"""
    files = os.listdir(folder_path)
    json_files = [folder_path+f for f in files if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('json')]
    return json_files

def load_json(pth):
    """加载json文件"""
    with open(pth, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data
def save_json(pth,data):
    """保存json文件"""
    with open(pth, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def shuffle_lists(list1, list2,seed=42):
    import random
    assert len(list1) == len(list2), "两个列表必须等长"
    random.seed(seed)
    # 创建索引列表
    indices = list(range(len(list1)))
    
    # 打乱索引列表
    random.shuffle(indices)
    
    # 使用打乱后的索引列表重新排列两个列表
    shuffled_list1 = [list1[i] for i in indices]
    shuffled_list2 = [list2[i] for i in indices]
    
    return shuffled_list1, shuffled_list2

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

def half_divide(img,data):
    """将图片从中分开,mask被穿过的char,并得到对应的左右json文件"""
    left_data={"shapes":[],"imageHeight":data["imageHeight"],"imageWidth":data["imageWidth"]//2}
    right_data={"shapes":[],"imageHeight":data["imageHeight"],"imageWidth":data["imageWidth"]//2}
        
    # 获取原始尺寸
    width, height = img.size
        
    # 计算切割点
    split_point = width // 2
    image_array = np.array(img)
    color,_=most_frequent_rgb(image_array)
    modified_image=image_array.copy()

    to_be_mask=[]
    for item in data['shapes']:
        if len(item['points'])!=2 or len(item['points'][0])!=2 or len(item['points'][1])!=2:
            continue
        [x1,y1],[x2,y2]=item['points']
        if x2<split_point:
            left_data['shapes'].append({"points":[[x1,y1],[x2,y2]]})
        elif x1>split_point:
            right_data['shapes'].append({"points":[[x1-split_point,y1],[x2-split_point,y2]]})
        else:
            to_be_mask.append([x1,y1,x2,y2])
        
    for coord in to_be_mask:
        x1, y1, x2, y2 = coord 
        modified_image[int(y1):int(y2), int(x1):int(x2)] =color  

    modified_image_pil = Image.fromarray(modified_image)
    left_img = modified_image_pil.crop((0, 0, split_point, height))
    right_img =modified_image_pil.crop((split_point, 0, width, height))
    return [left_img,left_data,right_img,right_data]

def refine(jpg_path,json_path,save_dir):
    """对一张图片进行half divide,直到子图都不超过300"""
    data=load_json(json_path)
    n=len(data['shapes'])
    name=jpg_path.split('/')[-1].split('.')[0]
    img = Image.open(jpg_path)
    if n<300:

        img.save(save_dir+name+f'.jpg')
        save_json(save_dir+name+f'.json',data)
        return None
    else:
        left_img,left_data,right_img,right_data=half_divide(img,data)
        ###储存所有当下的子图和子data
        sub_img=[left_img,right_img]
        sub_data=[left_data,right_data]
        i=0
        while True:
            if i==len(sub_img):
                break
            simg=sub_img[i]
            sdata=sub_data[i]
            if len(sdata['shapes'])>=300:
                sub_img.pop(i)
                sub_data.pop(i)
                li,ld,ri,rd=half_divide(simg,sdata)
                sub_img.append(li)
                sub_img.append(ri)
                sub_data.append(ld)
                sub_data.append(rd)             
                i-=1  
            i+=1
        j=0
        for pic,d in zip(sub_img,sub_data):
            save_json(save_dir+name+f'_{j}.json',d)
            pic.save(save_dir+name+f'_{j}.jpg')
            j+=1

def get_union(b1,b2):
    """求box之间的union,用于合并得列"""
    x1,y1,x2,y2=b1[0][0],b1[0][1],b1[1][0],b1[1][1]
    x3,y3,x4,y4=b2[0][0],b2[0][1],b2[1][0],b2[1][1]
    x=min(x1,x2,x3,x4)
    X=max(x1,x2,x3,x4)
    y=min(y1,y2,y3,y4)
    Y=max(y1,y2,y3,y4)
    return [[x,y],[X,Y]]
def list_union(boxes):
    """求一个box列表的union,得这列的box"""
    result=boxes[0]
    for item in boxes[1:]:
        result=get_union(result,item)
    return result
def get_col_jsons(json_files,jpg_files,base,destination_jpgs):
    """从gen_data转换为col_data,注意不是构建数据集,而是对每个json从字得列重新储存"""
    for file_path,jpg_path in tqdm(zip(json_files,jpg_files)):

        os.makedirs(destination_jpgs, exist_ok=True)

        # 构建源文件的完整路径
        source_file_path = os.path.join(base, jpg_path)
        
        # 构建目标文件的完整路径
        destination_file_path = os.path.join(destination_jpgs, jpg_path)
        
        # 复制文件到目标文件夹
        shutil.copy2(source_file_path, destination_file_path)

        i=file_path.split('.')[0]
        with open(base+file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        height=data["imageHeight"]
        width=data["imageWidth"]
        content=data['shapes']
        info=[]
        dic={}
        results=[]
        for item in content:
            col=item['col']
            if col not in dic:
                dic[col]=[item['points']]
            else:
                dic[col].append(item['points'])
        for key,value in dic.items():
            union=list_union(value)
            results.append({'label':key,'points':union})
        data['shapes']=results
        save_json(os.path.join(destination_jpgs,file_path ),data)
def drawBoxes(results,jpg_path,save_path):
    frame = cv2.imread(jpg_path)
    for points in results:
        x1, y1, x2, y2 = int(points[0][0]), int(points[0][1]), int(points[1][0]), int(points[1][1])
        cv2.rectangle(frame, (x1, y1), (x2, y2), thickness=2,color=(255,0,0),lineType=cv2.LINE_AA)
        label_position = ((x1+x2)//2,(y1+y2)//2)  # Adjust the position of the label as needed
        #cv2.putText(frame, str(idx), label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
    name=jpg_path.split("/")[-1]
    cv2.imwrite(save_path+"ordered_"+name,frame)


def intersection_length(x1, x3, x2, x4):
    # 计算两个区间的交集起始点和结束点
    start = max(x1, x2)
    end = min(x3, x4)

    # 如果交集起始点小于结束点，说明有交集
    if start < end:
        return end - start
    else:
        return 0


def union_length(x1, x3, x2, x4):
    # 计算并集起始点和结束点
    start = min(x1, x2)
    end = max(x3, x4)

    # 计算并集长度
    union_len = end - start

    return union_len


def distance_or_intersection(x1, x3, x2, x4):
    # 计算不相交两个区间的最短距离
    distance = min(abs(x1 - x4), abs(x2 - x3))

    # 判断是否相交
    if intersection_length(x1, x3, x2, x4) > 0:
        return 0  # 区间相交，返回0
    else:
        return distance  # 区间不相交，返回最短距离


def union(p1, p2):
    [x1, y1], [x2, y2] = p1
    [x3, y3], [x4, y4] = p2
    lx = min(x1, x3)
    ly = min(y1, y3)
    rx = max(x2, x4)
    ry = max(y2, y4)
    return [[lx, ly], [rx, ry]]

def merge_boxes(boxes,thresx=0.7, thresy=2):

    
    boxes = sorted(boxes, key=lambda box: (box[0][1]+box[1][1])/2)
    
    now_len=len(boxes)
    for _ in range(10):
        ydis_mean = 0
        for item in boxes:
            [x1, y1], [x3, y3] = item
            ydis_mean += abs(y1 - y3)
        length = len(boxes)
        if length==0:
            break
        ydis_mean /= length
        i = 0
        while i < length:
            j = 0
            # 依次遍历除自身外的全部box
            while j < length:
                mainbox = boxes[i]
                if i == j:
                    j += 1
                    continue
                length = len(boxes)
                # 算x区间上相交的程度
                intersection = intersection_length(mainbox[0][0], mainbox[1][0], boxes[j][0][0], boxes[j][1][0])
                x_rate = intersection / min(abs(mainbox[0][0] - mainbox[1][0]), abs(boxes[j][0][0] - boxes[j][1][0]))

                # 算y区间上相远离的程度,使用与字的y间距大小平均值的比值
                y_dis = distance_or_intersection(boxes[i][0][1], boxes[i][1][1], boxes[j][0][1], boxes[j][1][1])
                y_rate = y_dis / ydis_mean
                h1=abs(boxes[i][0][0]-boxes[i][1][0])
                h2=abs(boxes[j][0][0]-boxes[j][1][0])
                l1=abs(boxes[i][0][1]-boxes[i][1][1])
                l2=abs(boxes[j][0][1]-boxes[j][1][1])
                s1=h1*l1
                s2=h2*l2

                y_rate=y_dis/((l1+l2)/2)
                #print(min(s1,s2)/max(s1,s2))
                if x_rate > thresx and y_rate < thresy:
                    rm = boxes[j]

                    u = union(mainbox, rm)
                    # 更新第boxes[i],删除被合并的boxes[j]
                    boxes[i] = u
                    boxes.remove(rm)
                    # 处理各个指标的改变
                    if j < i:
                        i -= 1
                    length -= 1
                    j -= 1
                j += 1
            i += 1
        if now_len==len(boxes):
            break
        now_len=len(boxes)
    return boxes

def merge_boxes_new(boxes):
    boxes = sorted(boxes, key=lambda box: (box[0][1]+box[1][1])/2)


def char2col(jpg_path,boxes):
    columns=merge_boxes(boxes.copy())
    img = cv2.imread(jpg_path)
    h, w, channels = img.shape

    results={"imageHeight":h,"imageWidth":w,"shapes":[{"points":col} for col in columns]}
    return results