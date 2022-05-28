import tensorflow as tf
import numpy as np
import cv2

width_size = 256
height_size = 256
channel_size = 3

img_size = (width_size, height_size, channel_size)

cell_num = 3
class_num = 3

anchor_num = 1
label_num = anchor_num * (5 + class_num)

epoch_num = 20000

loss_p_rate = 1.0
loss_cod_rate = 5.0
loss_c_rate = 1.0
loss_p_no_rate = 0.5

def make_img_label():
    img = np.zeros((height_size+400,width_size+400,channel_size))
    label = np.zeros((cell_num,cell_num,label_num))
    num_shape = np.random.randint(1,4)
    i = np.random.choice(range(cell_num),num_shape,replace=False)
    j = np.random.choice(range(cell_num),num_shape,replace=False)
    
    img_0 = cv2.imread('0.png')
    img_1 = cv2.imread('1.png')
    img_2 = cv2.imread('2.png')
    
    for n_h in range(num_shape):
        row = i[n_h]
        col = j[n_h]
        
        shape_type = np.random.randint(0,class_num)
        x_rate = np.random.rand()
        y_rate = np.random.rand()
        w_rate = np.random.rand() * 0.3 + 0.1
        h_rate = np.random.rand() * 0.3 + 0.1
                
        label[row,col]=[1,x_rate,y_rate,w_rate,h_rate,0,0,0]
        label[row,col,5+shape_type]=1

        x = int(x_rate * width_size/cell_num + col * width_size/cell_num)
        y = int(y_rate * height_size/cell_num + row * height_size/cell_num)
        w = int(w_rate * width_size/2) * 2
        h = int(h_rate * height_size/2) * 2

        print(x, y, w, h)
        if(shape_type==0):
            input_img = cv2.resize(img_0,(w,h))
        if(shape_type==1):
            input_img = cv2.resize(img_1,(w,h))
        if(shape_type==2):
            input_img = cv2.resize(img_2,(w,h))

        img[y-int(h/2)+200 : y+int(h/2)+200, x-int(w/2)+200 : x+int(w/2)+200]=input_img

    img = img[200:200+height_size,200:200+width_size]        
    
    return img,label

img, label = make_img_label()

def cv2_imshow(a):
  """
  A replacement for cv2.imshow() for use in Jupyter notebooks.
  Args:
    a : np.ndarray. shape (N, M) or (N, M, 1) is an NxM grayscale image. shape
      (N, M, 3) is an NxM BGR color image. shape (N, M, 4) is an NxM BGRA color
      image.
  """
  a = a.clip(0, 255).astype('uint8')
  # cv2 stores colors as BGR; convert to RGB
  if a.ndim == 3:
    if a.shape[2] == 4:
      a = cv2.cvtColor(a, cv2.COLOR_BGRA2RGBA)
    else:
      a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
  
  cv2.imshow('image', a)

cv2_imshow(img)
cv2.waitKey(0)

def show_box(img, label, th=0.3):
    b_img = np.zeros((height_size+400, width_size+400, 3))
    b_img[200:200+height_size, 200:200+width_size] = img
    for i in range(cell_num):
        for j in range(cell_num):
            if(label[i, j, 0] > th):
                x_rate = label[i, j, 1]
                y_rate = label[i, j, 2]
                w_rate = label[i, j, 3]
                h_rate = label[i, j, 4]
                shape_type = np.argmax(label[i, j, 5:])
                if(shape_type==0):
                    line_color = [0, 0, 255]
                if(shape_type==1):
                    line_color = [255, 0, 0]
                if(shape_type==2):
                    line_color = [0, 255, 0]
                x = int(x_rate * width_size/3 + j * width_size/3)
                y = int(y_rate * height_size/3 + i * height_size/3)
                w = int(w_rate * width_size/2) * 2 + 20
                h = int(h_rate * height_size/2) * 2 + 20
                cv2.rectangle(b_img, (x-int(w/2)+200,y-int(h/2)+200), (x+int(w/2)+200,y+int(h/2)+200), line_color)
            
    b_img = b_img[200:200+height_size, 200:200+width_size]
    return b_img

cv2_imshow(show_box(img, label))
cv2.waitKey(0)

vgg_model = tf.keras.applications.VGG16(include_top=False, input_shape=img_size)
vgg_model.trainable=False

i = tf.keras.Input(shape=img_size)
out = tf.keras.layers.Lambda((lambda x : x/255.))(i)
out = vgg_model(out)
out = tf.keras.layers.Conv2D(256, 3, padding='same')(out)
out = tf.keras.layers.Conv2D(128, 3, padding='same')(out)
out = tf.keras.layers.Conv2D(64, 3, padding='same')(out)
out = tf.keras.layers.Flatten()(out)
out = tf.keras.layers.Dense(1024, activation='relu')(out)
out = tf.keras.layers.Dense(3*3*8,activation='sigmoid')(out)
out = tf.keras.layers.Reshape((3,3,8))(out)

yolo_model = tf.keras.Model(inputs=[i], outputs=[out])
opt = tf.keras.optimizers.Adam(0.00001)

yolo_model.summary()

fcc=cv2.VideoWriter_fourcc(*'DIVX')
out=cv2.VideoWriter('hjk_yolo.avi',fcc,1.0,(width_size,height_size))

for e in range(epoch_num):
    img,label = make_img_label()
    img = np.reshape(img,(1,height_size,width_size,3))
    label = np.reshape(label,(1,3,3,8))
    loss_p_list=[]
    loss_cod_list = []
    loss_c_list = []
    loss_p_no_list = []
    with tf.GradientTape() as tape:
        pred = yolo_model(img)
        # 이미지를 구분한 셀을 탐험
        for i in range(3):
            for j in range(3):
                # 해당 셀에 객체가 있을 경우는 확률, 박스 크기, 클래스까지 모두 Loss로 계산
                if(label[0,i,j,0]==1):
                    loss_p_list.append(tf.square(label[0,i,j,0]-pred[0,i,j,0]))
                    loss_cod_list.append(tf.square(label[0,i,j,1]-pred[0,i,j,1]))
                    loss_cod_list.append(tf.square(label[0,i,j,2]-pred[0,i,j,2]))
                    loss_cod_list.append(tf.square(label[0,i,j,3]-pred[0,i,j,3]))
                    loss_cod_list.append(tf.square(label[0,i,j,4]-pred[0,i,j,4]))
                    loss_c_list.append(tf.square(label[0,i,j,5]-pred[0,i,j,5]))
                    loss_c_list.append(tf.square(label[0,i,j,6]-pred[0,i,j,6]))
                    loss_c_list.append(tf.square(label[0,i,j,7]-pred[0,i,j,7]))
                # 해당 셀에 객체가 없을 경우 객체가 없을 확률만 Loss로 계산
                else:
                    loss_p_no_list.append(tf.square(label[0,i,j,0]-pred[0,i,j,0]))
        loss_p=tf.reduce_mean(loss_p_list)
        loss_cod =tf.reduce_mean(loss_cod_list)
        loss_c = tf.reduce_mean(loss_c_list)
        loss_p_no = tf.reduce_mean(loss_p_no_list)
        # 각 Loss를 비중을 곱해 더해 최종 Loss를 계산
        loss = loss_p_rate * loss_p + loss_cod_rate * loss_cod + loss_c_rate * loss_c + loss_p_no_rate * loss_p_no
    # Loss에 대한 Grad를 구하고, 각 파라미터를 업데이트
    vars = yolo_model.trainable_variables
    grad = tape.gradient(loss, vars)
    opt.apply_gradients(zip(grad, vars))
    # 100번 마다 동영상에 이미지를 기록한다
    if(e%100==0):
        img = np.reshape(img,(256,256,3))
        label = pred.numpy()
        label = np.reshape(label,(3,3,8))
        sample_img = np.uint8(show_box(img,label))
        out.write(sample_img)
    print(e,"완료",loss.numpy())    
out.release()