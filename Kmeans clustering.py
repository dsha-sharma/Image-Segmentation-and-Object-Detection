

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2

img = cv2.imread('a.jpg') 
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
original_shape = img.shape
print(original_shape)
plt.imshow(img)
plt.show()

flat_pixels = img.reshape((-1,3)) 
print(flat_pixels.shape)

from sklearn.cluster import KMeans

dom_colors = 5
kmeans = KMeans(n_clusters=dom_colors)
kmeans.fit(flat_pixels)

centers=kmeans.cluster_centers_

centers = np.array(centers,dtype='uint8')
print(centers) 

i = 1
plt.figure(0,figsize=(8,4))

colors = []
for each_color in centers:
    plt.subplot(1,5,i)
    plt.axis("off")
    i+=1
    colors.append(each_color)
    a = np.zeros((100,100,3),dtype='uint8')
    a[:,:,:] = each_color
    plt.imshow(a)

plt.show()

img2show = centers[kmeans.labels_]
segmented_img = img2show.reshape(original_shape)
plt.imshow(segmented_img)
plt.title("Segmented Image")
plt.axis("off")
plt.show()


fig = plt.figure(figsize=(20, 15))
ax1 = fig.add_subplot(1, 2, 1)
ax1.set_title("Original Image")
ax1.set_xticks([])
ax1.set_yticks([])
ax1.imshow(img)
ax2 = fig.add_subplot(1, 2, 2)
ax2.set_title("Segmented Image")
ax2.set_xticks([])
ax2.set_yticks([])
ax2.imshow(segmented_img)
plt.show()
