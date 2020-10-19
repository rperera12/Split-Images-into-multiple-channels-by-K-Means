import cv2
import math
import numpy as np 
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
#from IPython.display import clear_output


###################################################################################################################
#################---Function to break separate colors of an image into a single class for each---##################
###################################################################################################################
 
def color_masking(img, Klusters):

#         Inputs:  
#         1) img      - Image for clustering
#         2) Klusters - Number of Clusters 

# Defining input data for clustering
    data = np.float32(img).reshape((-1, 3))

# Defining criteria (See OpenCV K-Means Manual)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
# Applying cv2.kmeans function (See OpenCV K-Means Manual)
    _ , label, center = cv2.kmeans(data, Klusters, None, criteria, 10, cv2.KMEANS_PP_CENTERS) 

# Get resultant color groups and set as result array of shape (Klusters x 3 = RGB)   
    center = np.uint8(center)

    new = np.copy(center)
    result = center[label.flatten()]


# Sort color groups from blue to red
    sortt = np.zeros([3,Klusters])
    
    sorttt = sorted(new,key=lambda x: x[0])
# Create separate channels for each Kluster, each with shape ([192*192, 3]) 
    channel = {}
    for i in range(Klusters):
        sortt[:,i] = np.copy(sorttt[i])
        channel["channel{0}".format(i)] = np.zeros([len(result), 3])

# Loop through each pixel of the clustered image (3 channels per pixel x 192*192 pixels) and assign the 3-RGB values
    check_v = np.zeros([3,1])
    for k in range(len(result)):
        check_v[0,0] = np.copy(result[k,0])
        check_v[1,0] = np.copy(result[k,1])
        check_v[2,0] = np.copy(result[k,2])

  

    # Separate into 1-channel/1-class per clustered color 
    # Note - each class will contain 3 RGB values defining the clustered color at the respective location in the image  
        for j in range(Klusters): 
            if check_v[0,0] == sortt[0,j] and check_v[1,0] == sortt[1,j] and check_v[2,0] == sortt[2,j]:
                channel["channel{0}".format(j)][k,0] = np.copy(result[k,0])
                channel["channel{0}".format(j)][k,1] = np.copy(result[k,1])
                channel["channel{0}".format(j)][k,2] = np.copy(result[k,2])



# Reshape channels & full resultant clustered image into image dimensions (192,192,3)
# Note - In this description the img dimensions are 192x192, but this can be changed as needed by user without changes to function 
    for m in range(Klusters):
        channel["channel{0}".format(m)] = (channel["channel{0}".format(m)]).reshape(192,192,3)

    result = result.reshape(img.shape)

    return result, channel 



###################################################################################################################
#################----Function to plot each class and resulting combined image of all classes-----##################
###################################################################################################################

def plot_masks(Height, Width, Klusters, Channels):

#         Inputs:  
#         1) Height   - Height of Clustered/Segmented Image
#         2) Width    - Width of Clustered/Segmented Image   
#         3) Klusters - Number of Clusters used
#         4) Channels - Dictionary w/resultant clustered colors 

    combined_ch = np.zeros([Klusters, Height, Width, 3])
    final_result = np.zeros([Klusters+1,Height,Width,3])
    for k in range(Klusters):
        ch = Channels["channel{0}".format(k)]
 
        
        ch_p = cv2.cvtColor(ch.astype('uint8') , cv2.COLOR_RGB2BGR)

        if 'google.colab' in str(get_ipython()):
            from google.colab.patches import cv2_imshow
            cv2_imshow(ch_p.astype('uint8'))
        else:
            cv2.imshow('Channel', ch_p.astype('uint8')) 
            #plt.imshow(ch_p.astype('uint8'))
            #plt.show()
            #cv2.destroyWindow()
            cv2.waitKey(2000)     
    
        combined_ch[k, :, :, :] = ch_p 
        final_result[k+1,:,:,:] = final_result[k,:,:,:] + combined_ch[k,:,:,:]
  
    if 'google.colab' in str(get_ipython()):
        from google.colab.patches import cv2_imshow
        cv2_imshow(final_result[Klusters,:,:,:].astype('float32'))
    else:
         cv2.imshow('Combined Image', final_result[Klusters,:,:,:].astype('uint8'))
         #plt.imshow(final_result[Klusters,:,:,:].astype('float32'))
         #plt.show()
         cv2.waitKey(2000)

    return final_result[Klusters,:,:,:].astype('float32')




