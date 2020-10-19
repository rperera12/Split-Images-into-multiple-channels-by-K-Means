import cv2
import math
import numpy as np 
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
#from IPython.display import clear_output

from utils import color_masking, plot_masks

image = np.array(Image.open('images/original.png'))


K = 13
result, channel = color_masking(image, K)
combined = plot_masks(192, 192, K, channel)