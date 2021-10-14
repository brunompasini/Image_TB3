#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max, canny


# In[2]:


img = cv2.imread('TestPR46.png', cv2.IMREAD_GRAYSCALE)
plt.imshow(img)


# In[3]:


edge1 = canny(img)
edge2 = canny(img,2)
edge3 = canny(img,3)

edge = [edge1, edge2, edge3]

for el in edge:
    plt.figure()
    plt.imshow(el)


# In[4]:


edge_cv = cv2.Canny(img,100,200)
plt.imshow(edge_cv)


# In[5]:


X , Y = edge2.shape
ang_sampling = 0.002

rho_max = np.hypot(X,Y)
rho = np.arange(-rho_max, rho_max, 1)

theta = np.arange(0, np.pi, ang_sampling)
sin_theta, cos_theta = np.sin(theta), np.cos(theta)

H = np.zeros((rho.size, theta.size))

for i in range(X):
    for j in range(Y):
        if img[i,j] != 0:
            R = i*cos_theta + j*sin_theta
            R = np.round(R+(rho.size/2)).astype(int)
            H[R, range(theta.size)] += 1
                
plt.imshow(H)


# In[6]:


plt.figure()

G = cv2.GaussianBlur(H, (5,5), 5)
maxima = peak_local_max(H, 5, threshold_abs=150, num_peaks=5)

plt.scatter(maxima[:,1], maxima[:,0], c='r')
plt.imshow(G)


# In[9]:


for i_rho, i_theta in maxima:
    print(rho[i_rho], theta[i_theta])
    
    a = np.cos([theta[i_theta]])
    b = np.sin([theta[i_theta]])
    
    y0 = a*rho[i_rho]
    x0 = a*rho[i_rho]
    
    y1 = int(y0 + 1000*(-b))
    x1 = int(x0 + 1000*(a))
    
    y2 = int(y0 - 1000*(-b))
    x2 = int(x0 - 1000*(a))
    
    cv2.line(img, (x1,y1), (x2,y2), (0,0,255), 2)
    
cv2.imshow('hough lines',img)


# In[ ]:





# In[7]:


rho_peaks = rho[maxima[:,0]]
theta_peaks = theta[maxima[:,1]]

print(rho_peaks, theta_peaks)

