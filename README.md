# Instruction-Guided Visual Masking

[[paper]](https://arxiv.org/abs/2405.19783) [[project page]](https://2toinf.github.io/IVM/)

### 🔥 IVM has been selected as outstanding paper at MFM-EAI workshop @ICML2024

## Introduction

We introduce Instruction-guided Visual Masking (IVM), a new versatile visual grounding model that is compatible with diverse multimodal models, such as LMM and robot model. By constructing visual masks for instruction-irrelevant regions, IVM-enhanced multimodal models can effectively focus on task-relevant image regions to better align with complex instructions. Specifically, we design a visual masking data generation pipeline and create an IVM-Mix-1M dataset with 1 million image-instruction pairs. We further introduce a new learning technique, Discriminator Weighted Supervised Learning (DWSL) for preferential IVM training that prioritizes high-quality data samples. Experimental results on generic multimodal tasks such as VQA and embodied robotic control demonstrate the versatility of IVM, which as a plug-and-play tool, significantly boosts the performance of diverse multimodal models.

![1716817940241](image/README/1716817940241.png)

<p align="center">
    <img src="./image/gif/duck_greenplate2.gif" width="19.5%"/>
    <img src="./image/gif/redcup_redplate1.gif" width="19.5%"/>
    <img src="./image/gif/redcup_redplate3.gif" width="19.5%"/>
    <img src="./image/gif/redcup_silverpan1.gif" width="19.5%"/>
    <img src="./image/gif/redcup_silverpan5.gif" width="19.5%"/>
</p>
<p align="center">
 <em> Duck on green plate</em> |  <em>Red cup on red plate</em>  |  <em>Red cup on red plate</em>  |  <em>Red cup on silver pan </em> |  <em>Red cup on silver pan</em>
</p>

## Content

* [Quick Start](#quick-start)
* [Model Zoo](#quick-start)
* [Datasets](#quick-start)

## Quick Start

### Install

1. Clone this repository and navigate to IVM folder

```bash
git clone https://github.com/2toinf/IVM.git
cd IVM
```

2. Install Package

```bash
conda create -n IVM
pip install -e .
```

### Usage

```python
from IVM import load
from PIL import Image

model = load(ckpt_path, type="lisa", low_gpu_memory = False) # Set `low_gpu_memory=True` if you don't have enough GPU Memory

image = Image.open("assets/demo.jpg")
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

## Model Zoo

*Coming Soon*

## Evaluation

### VQA-type benchmarks

*Coming Soon*

### Real-Robot

Policy Learning: [https://github.com/Facebear-ljx/BearRobot](https://github.com/Facebear-ljx/BearRobot)

Robot Infrastructure: [https://github.com/rail-berkeley/bridge_data_robot](https://github.com/rail-berkeley/bridge_data_robot)

## Dataset

*Coming Soon*

## Acknowledgement

This work is built upon the [LLaVA](https://github.com/haotian-liu/LLaVA) and [SAM](https://github.com/facebookresearch/segment-anything). And we borrow ideas from [LISA](https://github.com/dvlab-research/LISA)

## Citation

```
@misc{zheng2024instructionguided,
            title={Instruction-Guided Visual Masking}, 
            author={Jinliang Zheng and Jianxiong Li and Sijie Cheng and Yinan Zheng and Jiaming Li and Jihao Liu and Yu Liu and Jingjing Liu and Xianyuan Zhan},
            year={2024},
            eprint={2405.19783},
            archivePrefix={arXiv},
            primaryClass={cs.CV}
        }
  
```
