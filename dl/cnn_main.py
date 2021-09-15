import cv2
from PIL import Image
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras.models import load_model,Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPool2D
from keras.callbacks import ModelCheckpoint,TensorBoard,Callback
from argparse import ArgumentParser

import random

f= open('basladi.txt',"w")
f.close()

parser = ArgumentParser()
parser.add_argument("-d","--data",default="data")
parser.add_argument("-m","--mod",default="hold")
parser.add_argument("-r","--renk",default="rgb")
args = parser.parse_args()
print(args)
klasor = args.data;

yols = os.listdir(klasor);
#print(data)

x_set=[]
y_set=[]
print("bilgi: resimler okunuyor...\n")
renk_uzay=cv2.COLOR_BGR2RGB
if(args.renk=="hsv"):
    renk_uzay=cv2.COLOR_BGR2HSV
elif(args.renk=="cie"):
    renk_uzay=cv2.COLOR_BGR2LAB


for yol in yols:
    data = os.listdir(klasor+"/"+yol)
    y=0
    print(yol)
    if(yol=="cloudy"):
        y=[1,0,0,0]
    elif(yol=="rainy"):
        y=[0,1,0,0]
    elif(yol=="shine"):
        y=[0,0,1,0]
    else:
        y=[0,0,0,1]
    print(y)
    for resim in data:
        rsm = cv2.imread(klasor+"/"+yol+"/"+resim)
        resim = cv2.cvtColor(rsm,renk_uzay)
        x_set.append(resim)
        y_set.append(y)

x_set = np.array(x_set)
y_set=np.array(y_set)
print("\n Veri seti bilgileri : ",x_set.shape,y_set.shape)

def karistir(x_set,y_set): #(c
    print("fonk")
    x_s=list(x_set)
    y_s=list(y_set)
    ras_x=[]
    ras_y=[]
    print(len(x_s),len(y_s))
    for i in range(0,x_set.shape[0]):
        ust=x_set.shape[0]-1-i;
        if(ust<=1):
            xv=x_s[ust]
            yv=y_s[ust]
            ras_x.append(xv)
            ras_y.append(yv)
            continue;
        sayi=random.randint(1,ust)
        xv=x_s[sayi]
        yv=y_s[sayi]
        x_s.pop(sayi)
        y_s.pop(sayi)
        ras_x.append(xv)
        ras_y.append(yv)
    return np.array(ras_x),np.array(ras_y)


x_veri,y_veri=karistir(x_set,y_set)

print("\n Karıstırılma Sonrası bilgiler : ",x_veri.shape,y_veri.shape)


batch_size=111
num_class=4
epoch=120
input_shape=(200,200,3)

# B
model = Sequential()
model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=input_shape)) 
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(256,(3,3),activation='relu'))
model.add(Conv2D(256,(3,3),activation='relu'))
model.add(Conv2D(256,(3,3),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(264,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(48,activation='relu'))
model.add(Dense(num_class,activation='softmax'));

model.summary() 

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer = keras.optimizers.Adadelta(),
              metrics=['accuracy'])
checkpoint = ModelCheckpoint("cnn_model_checkpoint.h5", monitor='acc', verbose=1, save_best_only=True, mode='max')

#{'loss': 1.5115363597869873, 'accuracy': 0.0, 'val_loss': 1.4188159704208374, 'val_accuracy': 0.0}
class Kayit(Callback):
    def on_epoch_end(self, epoch, logs=None):
        #print("epoch",logs)
        st = str(logs["loss"])+","+str(logs["accuracy"])+","+str(logs["val_loss"])+","+str(logs["val_accuracy"])+"\n"
        with open("kayit.txt","a") as f:
            f.write(st)
            


print("\n Bilgi: Eğitim Gerçekleştiriliyor \n")
if(args.mod=="hold"): #(e1
    x_test=x_veri[:220]
    y_test=y_veri[:220]
    x_train=x_veri[220:]
    y_train=y_veri[220:]
    print("\n x_test={}, y_test={}, x_train={}, y_train={}".format(x_test.shape,y_test.shape,x_train.shape,y_train.shape))
    sonuc = model.fit(x_train,y_train,batch_size=batch_size,epochs=epoch,verbose=1,validation_data=(x_test,y_test),callbacks=[checkpoint,Kayit()])
else: #(e2
    x_veri=x_veri[:1080]
    y_veri=y_veri[:1080]
    print("\n x_veri={}, y_veri={} ".format(x_veri.shape,y_veri.shape))

    x_veri=x_veri.reshape(12,90,200,200,3)
    y_veri=y_veri.reshape(12,90,4)
    print("\n x_veri={}, y_veri={} ".format(x_veri.shape,y_veri.shape))

    for ep in range(1,epoch+1):
        isim="epoch"+str(ep)+".h5"
        if(ep==1):
            model=model
        else:
            model = load_model("epoch"+str(ep-1)+".h5")
        dg = int(ep%12)

        if(dg==11):
            x_test=x_veri[dg]
            y_test=y_veri[dg]
            x_train=x_veri[0]
            y_train=y_veri[0]
        else:
            x_test=x_veri[dg]
            y_test=y_veri[dg]
            x_train=x_veri[dg+1]
            y_train=y_veri[dg+1]

        #print("\n x_test={}, y_test={}, x_train={}, y_train={}".format(x_test.shape,y_test.shape,x_train.shape,y_train.shape))

        print("\n Epoch:",dg);
        sonuc = model.fit(x_train,y_train,batch_size=1,epochs=1,verbose=1,validation_data=(x_test,y_test),callbacks=[Kayit()])
        model.save(isim)

    
model.save('cnn_model.h5')
sys.exit()
#plots(sonuc)


