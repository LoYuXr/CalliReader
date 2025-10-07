import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

def load_json(pth):
    with open(pth, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data
model = SentenceTransformer('all-MiniLM-L6-v2')

def compute_similarity(sentence1, sentence2):
    # 获取句子的嵌入向量
    embedding1 = model.encode(sentence1)
    embedding2 = model.encode(sentence2)
    # 计算余弦相似度
    return cosine_similarity([embedding1], [embedding2])[0][0]

src="outputs/exp/bilingual.json"
data=load_json(src)
sim=0
cnt=0
for item in tqdm(data):
    try:
        sentence = item["answer"].split("ENGLISH:")[-1].split("English:")[-1]
        similarity=compute_similarity(sentence, item['gt'])
        sim+=similarity
    except:
        sim+=0
    cnt+=1
print("STScore:",sim/cnt)
