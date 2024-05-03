## Quick Start

### Install

1. Clone this repository and navigate to SeeWhatYouNeedDeploy folder

```bash
git clone https://github.com/2toinf/SeeWhatYouNeedDeploy.git
cd SeeWhatYouNeedDeploy
```

2. Install Package

```bash
conda create -n CUNeed
pip install -e .
```

### Usage

```python
from deploy import load
model = load(ckpt_path, type="lisa", low_gpu_memory = False) # Set `low_gpu_memory=True` if you don't have enough GPU Memory

image = Image.open("assets/demo,jpg")
instruction = "pick up the red cup"

result = model(image, instruction, threshold = 0.5)
'''
result content:
	    'soft': heatmap
            'hard': segmentation map
            'blur_image': rbg
            'highlight_image': rbg
            'cropped_blur_img': rgb
            'cropped_highlight_img': rgb
            'alpha_image': rgba
'''




```
