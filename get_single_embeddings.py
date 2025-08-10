import sys
import os
from fontTools.ttLib import TTFont
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir,'models'))
sys.path.append(os.path.join(script_dir,'config'))
import opencc
cc = opencc.OpenCC('t2s.json')
import numpy as np
from PIL import Image, ImageOps, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from models.get_embeds import *
from models.model import *
from config.configu import *
import torch
import torch.nn.functional as F


DEVICE = "cuda:0"  # Set device to GPU 0


# Function to find the closest aspect ratio for resizing
def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

# Function to dynamically preprocess the image and split it into blocks
def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # Calculate the target ratios for resizing
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # Find the closest aspect ratio for the image
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # Calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # Resize the image to the target size
    resized_img = image.resize((target_width, target_height))
   
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # Split the image into blocks
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    
    assert len(processed_images) == blocks
    
    # If thumbnail is requested, add a small resized image
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    
    return processed_images

# Function to load and preprocess the image into pixel values
def load_image_2(image, input_size=448, max_num=12):
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    
    width, height = image.size
    
    # Resize the image based on its dimensions
    if max(width, height) <= 200:
        scale_factor = 200 / max(width, height)
    elif max(width, height) >= 350:
        scale_factor = 350 / max(width, height)
    else:
        scale_factor = 1.0

    # Resize image
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    image = image.resize((new_width, new_height))

    # Pad the image to the target input size
    padded_image = ImageOps.expand(image, border=(
        (input_size - new_width) // 2,    # Left padding
        (input_size - new_height) // 2,   # Top padding
        (input_size - new_width + 1) // 2, # Right padding
        (input_size - new_height + 1) // 2 # Bottom padding
    ), fill=(255, 255, 255))  # Fill with white color

    # Apply transformations
    transform = build_transform(input_size=input_size)

    # Preprocess image and return pixel values
    images = dynamic_preprocess(padded_image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    
    return pixel_values

# Function to calculate cosine similarity between input tensor and embedding
def vq_cos_sim(embedding, input_tensor, use_dynamic_p=False, ddp=False):
    if ddp:
        embedding_weight = embedding.module.weight
    else:
        embedding_weight = embedding.weight  # Shape: (N, d)

    # Normalize the tensors
    input_norm = F.normalize(input_tensor, p=2, dim=2)  # Normalize along dimension d
    embedding_norm = F.normalize(embedding_weight, p=2, dim=1)  # Normalize along dimension d

    # Calculate cosine similarity using matrix multiplication
    similarity = torch.matmul(input_norm, embedding_norm.t())  # Shape: (B, L, N)

    # Find the most similar embedding vector for each input
    cos_sim_values, indices = similarity.max(dim=2)

    if use_dynamic_p:
        # Return the closest vector indices and corresponding cosine similarity values
        return indices.squeeze(), cos_sim_values.squeeze()
    
    # Only return the closest vector indices
    return indices.squeeze()

# Function to get visual embeddings using a vision transformer
@torch.no_grad()
def get_visual_embed(pixel_values, vit, mlp1):
    vit_embeds = vit(
        pixel_values=pixel_values,
        output_hidden_states=False,
        return_dict=True
    ).last_hidden_state[:, 1:, :]  # Remove [CLS] token (b, 1025-1, 1024)

    # Adjust the shape
    h = w = int(vit_embeds.shape[1] ** 0.5)
    vit_embeds = vit_embeds.view(vit_embeds.shape[0], h, w, -1)

    # Downsample the embeddings
    vit_embeds = pixel_shuffle(vit_embeds, scale_factor=DOWNSAMPLE_RATIO)
    vit_embeds = vit_embeds.view(vit_embeds.shape[0], -1, vit_embeds.shape[-1])

    # Process with MLP
    vit_embeds = mlp1(vit_embeds)
    
    return vit_embeds

# Function to get visual embeddings for a single sample
@torch.no_grad()
def get_only_resampler(jspath, jpgpath, resampler, vit, mlp1, tok_embedding, mu, sigma, ddp=False):
    data = load_json(jspath)
    img = Image.open(jpgpath).convert("RGB")
    pixel_values = []
    cleaned_txt = ''
    h = data["imageHeight"]
    w = data['imageWidth']

    # Preprocess image regions and extract text labels
    for item in data['shapes']:
        cleaned_txt += cc.convert(item['label'])
        [x1, y1], [x2, y2] = item['points']
        x1 = min(max(0, x1), 1)
        y1 = min(max(0, y1), 1)
        y2 = min(max(0, y2), 1)
        x2 = min(max(0, x2), 1)
        x1 = int(x1 * w)
        x2 = int(x2 * w)
        y1 = int(y1 * h)
        y2 = int(y2 * h)

        try:
            orig_width, orig_height = Image.fromarray(np.array(img)[y1:y2, x1:x2]).size
            if orig_height * orig_width == 0:
                import time
                print(x1, y1, x2, y2, item['label'], jspath)
                time.sleep(10030)
                
            sub_img = Image.fromarray(np.array(img)[y1:y2, x1:x2])
        except:
            continue
        
        pixel_value = load_image_2(sub_img).to(torch.bfloat16).cuda()
        pixel_values.append(pixel_value)
    
    pixel_values = torch.cat(pixel_values, dim=0)
    visual_embed = get_visual_embed(pixel_values, vit, mlp1)
    output = resampler(visual_embed)

    # Calculate cosine similarity
    outs = vq_cos_sim(tok_embedding, output, ddp=ddp)
    indices = outs

    flattened_output = output.view(-1, output.shape[-1])
    flattened_indices = indices.view(-1)

    filtered_indices = flattened_indices[flattened_indices != 0]
    filtered_output = flattened_output[flattened_indices != 0]        

    sigma_flat = sigma[filtered_indices]
    mu_flat = mu[filtered_indices]

    sigma_flat = sigma_flat.expand(-1, filtered_output.shape[-1])
    mu_flat = mu_flat.expand(-1, filtered_output.shape[-1])
    back_to_origin_flat = filtered_output * sigma_flat + mu_flat

    return back_to_origin_flat.detach().cpu()

# Function to extract the embedding for a single sample and save it
@torch.no_grad()
def extract_single_embedding(jspath, jpgpath, out_path_pt):
    """
    Generate embeddings for a single (json, jpg) sample and save them to out_path_pt (.pt) or out_path_p (.p)
    Provide one or both output paths as needed.
    """
    # 1) Load models (only load once for efficiency)
    resampler = load_perceiver_resampler_2(
       './params/callialign.pth',
        num_layers=4
    ).to(DEVICE)
    vit = load_vision_model().to(DEVICE)
    mlp1 = load_mlp1(downsample_ratio=DOWNSAMPLE_RATIO).to(DEVICE)
    tok_embedding = load_normed_tok_embeddings().to(DEVICE)
    _ = load_tok_embeddings().to(DEVICE)

    # 2) Load normalization parameters
    mu_sigma = torch.load("./params/gauss_norm_mu_sigma.pth")
    mu = mu_sigma['weight'][:, 0].reshape((-1, 1)).to(DEVICE)
    sigma = mu_sigma['weight'][:, 1].reshape((-1, 1)).to(DEVICE)

    # 3) Calculate embedding
    emb = get_only_resampler(
        jspath=jspath,
        jpgpath=jpgpath,
        resampler=resampler,
        vit=vit,
        mlp1=mlp1,
        tok_embedding=tok_embedding,
        mu=mu,
        sigma=sigma,
        ddp=False
    )

    # 4) Save embedding

    os.makedirs(os.path.dirname(out_path_pt), exist_ok=True)
    torch.save(emb, out_path_pt)

    return emb

if __name__ == "__main__":
    # Example usage: Provide JSON and JPG paths
    jspath = "./examples/0.json"
    jpgpath ="./examples/0.jpg"

    # Output paths for saving embeddings
    out_pt = "./results/embedding.pt"

    emb = extract_single_embedding(jspath, jpgpath, out_path_pt=out_pt)
    print("Embedding shape:", emb.shape)
    print("Saved to:", out_pt)
