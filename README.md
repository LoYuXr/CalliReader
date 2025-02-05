# CalliReader
Official repository for CalliReader: Deciphering Chinese Calligraphy via an Embedding-aligned Vision Language Model

## Inference
Please first download the params.zip and unzip it to get a ```params``` folder, and download finetuned InternVL model weights into ```InternVL``` folder.

For a single image, use
```
python inference.py --tgt=<image path> 
```

For a folder with multiple images, use
```
python inference.py --tgt=<folder path>  --save_name=<your save name>
```
and results will be saved to ```<your save name>.json```.
