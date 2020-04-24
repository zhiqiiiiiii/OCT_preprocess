This repository is adapted from [Weiting Tan](https://github.com/steventan0110/OCT_preprocess)
* Item read_img.py: read Cirrus .img file to numpy array.
* Item detect_retina: detect boundaries for ILM, IS/OS, and BM. Then the volumn is flatterned to BM.
* Item inpaint_nan3.py: fill in values for nan points. 

Usage:
```python
from read_img import read_img
from detect_retina import detect_retina 
img_vol = read_img(filepath, pixelsX, pixelsY, pixelsZ)
vol_flatterned, retina_mask, upper_bound, lower_bound = detect_retina(img_vol)
```
