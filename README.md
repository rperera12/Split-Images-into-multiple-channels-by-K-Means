# Split-Images-into-multiple-channels-by-K-Means
By K-Means Clustering using OpenCV-Python an image can be broken up or split into multiple channels (user defined), where each channel represents a single color group extracted from the image. The main purpose of this repository is to create a mask of 13 channels from already segmented images. However, it can be used for other applications such as simple color extraction. 

```bash
pip install foobar
```

```
python
import cv2
import math
import numpy as np 
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from utils import color_masking, plot_masks

image = np.array(Image.open('images/original.png'))

K = 13
result, channel = color_masking(image, K)
combined = plot_masks(192, 192, K, channel)
```
