<h2 align="center">
  <b>CalliReader<img src="imgs\logo.png" alt="Image" width="30" height="30">: Contextualizing Chinese Calligraphy via
an Embedding-aligned Vision Language Model</b>



<div align="center">
     <a href="https://arxiv.org/abs/2503.06472" target="_blank">
        <img src="https://img.shields.io/badge/Paper-ArXiv-red" alt="Paper arXiv">
    </a>
    <img src="https://img.shields.io/badge/Page-CalliReader-blue" alt="Project Page"/></a>
    <a href="your pdf here" target="_blank">
    <img src="https://img.shields.io/badge/Lab-Link-green" alt="Lab Link"></a>
</div>
</h2>

This is the repository of [**CalliReader: Contextualizing Chinese Calligraphy via
an Embedding-aligned Vision Language Model**](https://arxiv.org/pdf/2503.06472).

CalliReader is a novel plug-and-play Vision-Language Model (VLM) specifically designed to interpret calligraphic artworks with diverse styles and layouts, leveraging slicing priors, embedding alignment, and effective fine-tuning. It demonstrates remarkable performances on Chinese Calligraphy recognition and understanding, while also retains excellent OCR ability on general scenes.

For more information, please visit our [**project page**](https://your_page_here/)    (Unfinished).

![teaser](imgs/teaser.jpg)

## 📬 News
- **2025.8.10** We are releasing our training scripts.
- **2025.6.26** Our newest model is now available on HuggingFace.
- **2025.6.25** Our work has been accepted by ICCV 2025!
- **2025.2.12** The repository has been updated.

## How to Use Our Code and Model:
We are releasing our network and checkpoints. You can download weights of our CalliReader from this [**HuggingFace  link**](https://huggingface.co/gtang666/CalliReader/tree/main). Finetuned VLM weight files that end with ```.safetensors``` are stored in the folder ```InternVL```, and all pluggable modules can be found in the folder ```params```. You can download those files and put them in the same folder of the cloned repository.

You can setup the pipeline under the following guidance.

### 0. Install dependencies
1. We recommend creating a conda environment with Python>=3.9 and activate it:
```
conda create -n callireader python=3.9
conda activate callireader
```
2. Then, install essential dependencies:
```
pip install requirements.txt
```
3. Finally, install the package ```flash-attn```:
```
pip install flash-attn
```
If you encounter certain problems with this package, you can download .whl file [here](https://github.com/Dao-AILab/flash-attention/releases) for direct installation:
```
pip install flash_attn-xxx.whl
```
Please note that this package only supports Linux systems with CUDA installed, and all of their versions should be matched. For further issues about ```flash-attn```, please turn to its [repository](https://github.com/Dao-AILab/flash-attention) for help. 

### 1. Inference
We have verified that ```.jpg``` and ```.png``` format images are well supported.

1. For a single image, use
```
python inference.py --tgt=<image path> 
```
The result will be output directly in the terminal.

2. For a folder with multiple images, use
```
python inference.py --tgt=<folder path>  --save_name=<your save name>
```
Results will be saved to ```./results/<your save name>.json```.

### 2. Dataset
#### Full-page Recognition
Data of three tiers (easy, medium, hard) of Full-page Recognition can be downloaded in this [**link**](https://huggingface.co/datasets/gtang666/CalliBench). It contains 3,192 samples in total. We will further update other components of our CalliBench.
#### Training data
Data of 7,357 samples for e-IT can be downloaded in this [**link**](https://huggingface.co/datasets/gtang666/CalliTrain).


### 3. Training
Please refer to the **train** folder for further instructions. 


## Citation
```
@article{luo2025callireader,
  title={CalliReader: Contextualizing Chinese Calligraphy via an Embedding-Aligned Vision-Language Model},
  author={Luo, Yuxuan and Tang, Jiaqi and Huang, Chenyi and Hao, Feiyang and Lian, Zhouhui},
  journal={arXiv preprint arXiv:2503.06472},
  year={2025}
}
```
