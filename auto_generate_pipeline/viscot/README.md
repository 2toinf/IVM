### Usage

1. Your prompt should refomulate as following:

*"`<image>`\n{`your instruction here`} Please provide the bounding box coordinate of the region that can help you answer the question better."*

2. Then you can get a bounding box output like this:

*[0.562, 0.228, 0.646, 0.292]*

3. After get the bounding box, you can generate the mask as following code:

```python
def generate_boxes_mask(xmin, ymin, xmax, ymax, height, width):
    mask = np.zeros((height, width), dtype=np.float32)
    mask[int(ymin * height):int(ymax * height), int(xmin * width):int(xmax * width)] = 1
    return mask
# Note that PIL.Image.size will return -> [width, height]
# generate_boxes_mask(*(item['boxes'] + [image.size[1], image.size[0]]))
```
