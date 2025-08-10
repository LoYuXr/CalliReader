## Training

### Setup
Our e-IT is built upon [Xtuner](https://github.com/InternLM/xtuner) with **CUDA VERSION=12.4** and **Ubuntu=22.04**. Run the following command to setup a training environment:
```
cd train
conda env create -f environment.yml
conda activate train
pip install xtuner==0.1.23
```
### Data Preparation
We process the raw training dataset and organize a JSON file for training. It should be a list of dict, and each item should be organized in the following format:
```
    {
        "id": xxx, # The index of the item
        "embedding": "xxx.pt", # The path to the pseudo-text embeddings of the image 
        "image": "/data/luoyx/finetuned_data/929_data/train_img/2137.jpg", # The image path
        "conversations": [
            {
                "from": "human",
                "value": "xxx" 
            },
            {
                "from": "gpt",
                "value": "xxx"
            }
        ],  # The converstaion for e-IT
        "rank": 0
    }
```
We provide an exmaple in this [link](https://drive.google.com/file/d/1433HhzZK8xQzK0pFsdoYH4SJjR72eTdf/view?usp=sharing) with 1000 images and pseudo-text embeddings. You can download and unzip it under the training folder. We use absolute paths, so you should replace the prefix of all paths.


#### Synthesizing Pseudo-text Embeddings
If you want to synthesize more pseudo-text embeddings, you can refer to **get_single_embeddings.py** in the main folder. It uses a JPG file and the corresponding JSON annotation to generate a .pt file.

### Run
Please modify the following variables in **./xtuner/configs/internvl/v2/e-IT.py** to be compatible with your custom path.
```
path = '/home/luoyx/InternVL/InternVL' # The path to the .safetensors files
data_root = '/home/luoyx/train/samples/' # The path to the data files
data_path = [data_root + 'samples.json']
```
Once the setup is complete, run the following command to start the training:
```
export PYTHONPATH="../InternVL" # The path to the .safetensors files
NPROC_PER_NODE=2 xtuner train ./xtuner/configs/internvl/v2/e-IT.py --work-dir ./logs/e_it --deepspeed deepspeed_zero1
```

Run the following command to merge the LoRA weights with InternVL:
```
export PYTHONPATH="../InternVL"
python ./xtuner/configs/internvl/v1_5/convert_to_official.py --config=./xtuner/configs/internvl/v2/e-IT.py ./logs/e_it/iter_xxx.pth /path/to/save/models
```

Final weights can be found in your **/path/to/save/models**, which can be used for further evaluations.

#### Notes
If there is any problem with paths, we recommend using absolute paths to ensure correctness.