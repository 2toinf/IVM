import numpy as np
from PIL import Image
import json
def generate_boxes_mask(xmin, ymin, xmax, ymax, height, width):
    mask = np.zeros((height, width), dtype=np.float32)
    xmin = xmin * max(width, height)
    ymin = ymin * max(width, height)
    xmax = xmax * max(width, height)
    ymax = ymax * max(width, height)
    if width > height:
        overlay = (width - height) // 2
        ymin = max(0, ymin - overlay)
        ymax = max(0, ymax - overlay)
    else:
        overlay = (height - width) // 2
        xmin = max(0, xmin - overlay)
        xmax = max(0, xmax - overlay)
    mask[int(ymin):int(ymax), int(xmin):int(xmax)] = 1
    return mask

image = Image.open(...)
item = json.loads(...)
probability_mask = generate_boxes_mask(*(item['boxes'] + [image.size[1], image.size[0]]))