import numpy as np
import zipfile
import matplotlib.pyplot as plt
from PIL import Image
import os
import tensorflow as tf
import cv2

def iris_mask(img):
	
    img_org_gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    img_histEqualization= cv2.equalizeHist(img_org_gray)
    _,img_binary= cv2.threshold(img_org_gray,70,255,cv2.THRESH_BINARY)
    center_pupil = cv2.HoughCircles(img_binary, cv2.HOUGH_GRADIENT,1,250,
                           param1=80,param2=10,minRadius=30,
                           maxRadius=70)
    center_pupil = np.uint16(np.around(center_pupil))
    draw = img.copy()
#     for i in center_pupil[0,:]:
#         print(i,i[0],i[1],i[2])
#         draw = cv2.circle(img,(i[0],i[1]),i[2],(255,0,0),2)
#         draw = cv2.circle(draw,(i[0],i[1]),2,(255,0,0),3)
    (x1, y1) = center_pupil[0][0][0], center_pupil[0][0][1]
    circles = cv2.HoughCircles(img_histEqualization, cv2.HOUGH_GRADIENT,1,250,
                           param1=80,param2=30,minRadius=50,
                           maxRadius=100)
    circles = np.uint16(np.around(circles))
    draw_shift = draw.copy()     
    for i in circles[0,:]:       
        print(i,i[0],i[1],i[2])
        mask = np.zeros_like(img)
        mask = cv2.circle(mask, (x1,y1), i[2], (255,255,255), -1)
        draw_shift = cv2.circle(draw_shift,(x1,y1),i[2],(255,0,0),2)
        #draw_shift = cv2.circle(draw_shift,(x1,y1),2,(255,0,0),3)  
        draw_mask = cv2.bitwise_and(draw_shift, mask)
    return draw_mask


x_train, y_train = np.zeros([2688, 224, 224,3]), np.zeros([2688])
x_test, y_test = np.zeros([896, 224, 224,3]), np.zeros([896])

folder = "IITD_database/people"
i = 0
for people in range(1,225):
    person_folder = folder + "/" + "0"*(3-len(str(people))) + str(people)
    file_names = os.listdir(person_folder)
    for num_image in range(1,7):
        path = person_folder + "/" + file_names[num_image-1]
        image = Image.open(path)
        image = image.resize((224,224))
        image = np.asarray(image)
        image = iris_mask(image)
        #x = image.reshape((1,224,224,3))
        row_num = i*6 + num_image -1
        x_train[row_num,:,:,:] = image
        y_train[row_num] = people
        
        flip = np.flip(x_train[i],1)
        flip = np.resize(flip, (1,224,224,3))
        x_train[row_num+1344,:,:,:] = flip
        y_train[row_num+1344] = people
    i += 1
    
i = 0
for people in range(1,225):
    person_folder = folder + "/" + "0"*(3-len(str(people))) + str(people)
    file_names = os.listdir(person_folder)
    for num_image in range(7,11):
        path = person_folder + "/" + file_names[num_image-1]
        image = Image.open(path)
        image = image.resize((224,224))
        image = np.asarray(image)
        image = iris_mask(image)
        #x = image.reshape((1,224,224,3))
        row_num = i*4 + num_image -7
        x_test[row_num,:,:,:] = image
        y_test[row_num] = people
    i += 1
    
x_train.shape, y_train.shape, x_test.shape, y_test.shape

index_train = np.random.permutation(2688)
index_test = np.random.permutation(896)
x_train, y_train = x_train[index_train], y_train[index_train]
x_test, y_test = x_test[index_test], y_test[index_test]

x_train, x_test = tf.cast(x_train, tf.float32)/255.0, tf.cast(x_test, tf.float32)/255.0
y_train, y_test = tf.one_hot(y_train-1, depth=224), tf.one_hot(y_test-1, depth=224)
#y_train, y_test = tf.cast(y_train, tf.float32), tf.cast(y_test, tf.float32)