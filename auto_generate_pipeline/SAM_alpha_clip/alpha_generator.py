from typing import IO
from sklearn.model_selection import ParameterSampler
import torch
import alpha_clip
from PIL import Image
import numpy as np
from torchvision import transforms
import argparse
import json
from mmengine import fileio
import io
import tqdm
import utils
parser = argparse.ArgumentParser()
parser.add_argument('--meta_file', type=str)
parser.add_argument('--image_root', type=str)
parser.add_argument('--mask_root', type=str)
parser.add_argument('--save_root', type=str)
parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
parser.add_argument('--port', default=29529, type=int, help='port')    
args = parser.parse_args()
utils.init_distributed_mode(args, verbose=False)
# load model and prepare mask transform
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = alpha_clip.load("ViT-L/14@336px", alpha_vision_ckpt_pth="/mnt/lustre/zhengjinliang/See-what-you-need-to-see/AlphaCLIP/clip_l14_336_grit_20m_4xe.pth", device=device)  # change to your own ckpt path
mask_transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Resize((336, 336)),
    transforms.Normalize(0.5, 0.26)
])

with open(args.meta_file, "r") as f:
    metas = f.readlines()
process_length = len(metas) / utils.get_world_size()
global_rank = utils.get_rank()
start_idx = int(global_rank * process_length)
end_idx = int((global_rank + 1) * process_length)
for idx in  tqdm.tqdm(range(start_idx, end_idx)):
    if fileio.exists(fileio.join_path(args.save_root, f'{idx}.npz')): continue
    meta = json.loads(metas[idx])
    value = fileio.get(fileio.join_path(args.image_root,meta['image']))
    img_bytes = np.frombuffer(value, np.uint8)
    buff = io.BytesIO(img_bytes)
    image = Image.open(buff).convert('RGB')
    try:
        image_name = meta['image'].split('/')[-1].split('.')[0]
        image_path = '/'.join(meta['image'].split('/')[:-1])
        mask = np.load(io.BytesIO(fileio.get(fileio.join_path(args.mask_root, image_path, f'{image_name}.npz'))))['arr_0']
    except:
        print(fileio.join_path(args.mask_root, image_path, f'{image_name}.npz'))
        continue
    num_mask = mask.max()
    image = preprocess(image).unsqueeze(0).repeat(num_mask+1, 1, 1, 1).half().to(device)
    binary_mask = []
    for i in range(num_mask+1):
        alpha = mask_transform(((mask == i) * 255).astype(np.uint8))
        binary_mask.append(alpha)
    binary_mask = torch.stack(binary_mask, dim = 0).half().cuda()
    text = alpha_clip.tokenize([meta['instruction']]).to(device)

    with torch.no_grad():
        image_features = model.visual(image, binary_mask)
        text_features = model.encode_text(text)

    # normalize
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    ## print the result
    similarity = (image_features @ text_features.T).squeeze() + 1
    similarity = (similarity - similarity.min()) / (similarity.max() - similarity.min()) 
    clip_score_map = np.zeros_like(mask, dtype=np.float32)
    for i in range(num_mask+1):
        clip_score_map[mask == i] = similarity[i].item()

    with io.BytesIO() as f:
        np.savez_compressed(f, clip_score_map)
        fileio.put(f.getvalue(), fileio.join_path(args.save_root, f'{idx}.npz'))
        
        
