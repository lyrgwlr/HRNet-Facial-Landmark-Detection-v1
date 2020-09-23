#!/usr/bin/env python
# coding: utf-8

# In[47]:


import numpy as np
import cv2
import os
import matplotlib.pyplot as plt


# In[50]:


ori_img_path = '/data/yabo.xiao/val/picture/'
out_img_path = '/data/yabo.xiao/val/out_vis/'
if not os.path.exists(out_img_path):
    os.makedirs(out_img_path) 


img_list = os.listdir(ori_img_path)
# import pudb; pudb.set_trace()
for img in img_list:

    a = cv2.imread(os.path.join(ori_img_path,img))
    landmarks = []
    preds = '/data/yabo.xiao/val/res/' + img.split('.')[0] + '.txt'
    with open(preds) as f:
        lines = f.readlines()
    for line in lines[1:]:
        x, y = line.strip().split(' ')
        x, y = int(float(x)), int(float(y))
        landmarks.append([x, y])



    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, lm in enumerate(landmarks):
        x, y = lm
        a = cv2.circle(a, (x, y), 1, (0, 255, 0), 4)
    save_path = out_img_path + img
    cv2.imwrite(save_path, a)





