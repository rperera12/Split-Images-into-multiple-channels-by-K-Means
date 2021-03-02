# Split-Images-into-multiple-channels-by-K-Means
By K-Means Clustering using OpenCV-Python an image can be broken up or split into multiple channels (user defined), where each channel represents a single color group extracted from the image. The main purpose of this repository is to create a mask of 13 channels from already segmented images. However, it can be used for other applications such as simple color extraction. 

# Code Example
The following shows an example of the functions. The same results can be generated by running test.py

```python
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
# Results 

### Original Segmented Image

[Original Segmented Image](/images/original.png)


### Separated Channels/Classes/Colors using Clusters = 13

[1st Channel Segmented Image](/images/channel1.png)

[2nd Channel Segmented Image](/images/channel2.png) 

[3rd Channel Segmented Image](/images/channel3.png) 

[4th Channel Segmented Image](/images/channel4.png) 

[5th Channel Segmented Image](/images/channel5.png) Format: [Channel 5](url)  [6th Channel Segmented Image](/images/channel6.png) Format: [Channel 6](url)

[7th Channel Segmented Image](/images/channel7.png) Format: [Channel 7](url)  [8th Channel Segmented Image](/images/channel8.png) Format: [Channel 8](url)

[9th Channel Segmented Image](/images/channel9.png) Format: [Channel 9](url)  [10th Channel Segmented Image](/images/channel10.png) Format: [Channel 10](url)

[11th Channel Segmented Image](/images/channel11.png) Format: ![Channel 11](url) [12th Channel Segmented Image](/images/channel12.png) Format: [Channel 12](url)

[13th Channel Segmented Image](/images/channel13.png) Format: [Channel 13](url)

### Finally, all the channels are combined together and displayed below, note & compare the original segmented image to the resultant image shown below

[Original Segmented Image](/images/original.png) Format: ![Original](url)  [Combined Results Image](/images/combined_results.png)
Format: [Combined Results](url)
