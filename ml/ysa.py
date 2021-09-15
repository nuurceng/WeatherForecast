import cv2
from PIL import Image
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from skimage.feature import daisy

import keras
from keras.models import load_model,Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

data = os.listdir('dataset2');
#print(data)

rgb=[]
hsv=[]
cie=[]
hata=[]
print("bilgi: resimler okunuyor...\n")
i=0
for resim in data:
    rsm = cv2.imread("dataset2/"+resim)
    
    #print(resim,"için =>")
    try:
        rgb_resim = cv2.cvtColor(rsm,cv2.COLOR_BGR2RGB)
        #hsv_resim = cv2.cvtColor(rsm,cv2.COLOR_BGR2HSV)
        #cie_resim = cv2.cvtColor(rsm,cv2.COLOR_BGR2LAB)
        rgb.append(rgb_resim)
        #hsv.append(hsv_resim)
        #cie.append(cie_resim)
    except:
        #print(resim,"hatali !!!");
        hata.append([resim,i])
    print(resim,"okundu")
    i+=1;



print("\nbilgi: hatalar temizleniyor..\n")
for sil in hata:
    data.pop(sil[1])




# RGB İÇİN
print("\nbilgi: resimler rgb formatta surf,sift öznitelikleri çıkartılıyor\n")


def surf_oznitelik(resim):
    surf = cv2.xfeatures2d.SURF_create()
    kp, des = surf.detectAndCompute(resim,None)
    return kp

def sift_oznitelik(resim):
    surf =cv2.xfeatures2d.SIFT_create()
    kp, des = surf.detectAndCompute(resim,None)
    return kp


print("\nbilgi: Öznitelikler Çıkartılıyor ...\n")
surfs=[]
#sifts=[]
boy=len(rgb)
i=1
for resim in rgb:
    print(i,"/",boy," kadar resmin özniteliği çıkartıldı")
    sf=surf_oznitelik(resim)
    #st=sift_oznitelik(resim)
    surfs.append(sf)
    #sifts.append(st)
    i+=1;




print("\nbilgi: X ve Y değerleri oluşturuluyor\n")
#100x100 lık alanda kac tane var ?
def x_girdi(oz):
    don=[]
    for i in range(1,7):
        satir_max=i*100
        satir_min=satir_max-100
        
        for j in range(1,5):
            sutun_max=j*100
            sutun_min=sutun_max-100
            oz_sayi=0
            for nitel in oz:
                x = nitel.pt[0]
                y = nitel.pt[1]
                if((x<=satir_max and y<=sutun_max) and (x>=satir_min and y>=sutun_min)):
                    oz_sayi+=1
             #print("(",satir_min,sutun_min,"),(",satir_max,sutun_max,") kısımları arasından ki öznitelik sayısı=>",oz_sayi);
            don.append(oz_sayi)
    return don # resmi 6*4=24 parcaya bolundu ve her parcada ne kadar olduğu yer işaretlendi


lst = [str(i) for i in range(1,25)]
# Label Map =>
#bulutlu (cloudy) = 1
#Yagmurlu (rain) =2
# gunesli (shine) = 3
# gun batımı (sunrise)=4
lst.append("y")
lst.insert(0,"dosya")
veri_kumesi = pd.DataFrame(columns=lst)
#surf ile

for oz,rsm in list(zip(surfs,data)):
    
    nitelik=x_girdi(oz)
    nitelik.insert(0,rsm)
    y=0
    
    if(rsm[0]=="c"):
        y=1
    elif(rsm[0]=="r"):
        y=2
    else:
        if(rsm[1]=="h"):
            
            y=3
        else:
            y=4
            
    nitelik.append(y)
    veri_kumesi.loc[len(veri_kumesi)]=nitelik
    print(rsm," için eğitim seti oluşturuldu")

print("\n Eğitim Seti => \n")
print(veri_kumesi.head(),"\n")


veri_x= veri_kumesi.drop(["dosya","y"],axis=1)
veri_y = veri_kumesi["y"]

x_train, x_test, y_train, y_test = train_test_split(veri_x,veri_y,test_size=0.2,random_state=10)

x_train=x_train.values.tolist()
x_test=x_test.values.tolist()
y_train=y_train.tolist()
y_test=y_test.tolist()

for i in range(len(y_train)):
    if(y_train[i]==1):
        y_train[i]=[1,0,0,0]
    elif(y_train[i]==2):
        y_train[i]=[0,1,0,0]
    elif(y_train[i]==3):
        y_train[i]=[0,0,1,0]
    elif(y_train[i]==4):
        y_train[i]=[0,0,0,1]

for i in range(len(y_test)):
    if(y_test[i]==1):
        y_test[i]=[1,0,0,0]
    elif(y_test[i]==2):
        y_test[i]=[0,1,0,0]
    elif(y_test[i]==3):
        y_test[i]=[0,0,1,0]
    elif(y_test[i]==4):
        y_test[i]=[0,0,0,1]


x_train=np.array(x_train)
x_test=np.array(x_test)
y_train=np.array(y_train)
y_test=np.array(y_test)

print("\n ogrenım oncesi => x_train:{}, x_test:{}, y_train:{}, y_test:{}\n".format(
    x_train.shape,
    x_test.shape,
    y_train.shape,
    y_test.shape))

batch_size=40
num_class=4
epoch=35
input_shape=(24,)

print(x_train[0])
model = Sequential()

#

print("\n Model Olusturuluyor \n")
model.add(Dense(96,activation='relu',input_shape=input_shape))
model.add(Dense(48,activation='relu'))
model.add(Dense(24,activation='relu'))
model.add(Dense(num_class,activation='softmax'))

model.summary()

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer = keras.optimizers.Adam(learning_rate=0.01),
              metrics=['accuracy'])

print("\n Eğitim Gerçekleşiyor \n")
model.fit(x_train,y_train,batch_size=batch_size,epochs=epoch,verbose=1,validation_data=(x_test,y_test))

model.save("ysa.h5")

bt={"1":0,"2":0,"3":0,"4":0}
sonuc = model.predict(x_test,batch_size=len(x_test))
yeni_s=[]
for i in sonuc:
    bt["1"]=i[0]
    bt["2"]=i[1]
    bt["3"]=i[2]
    bt["4"]=i[3]
    fz=max(bt,key=bt.get)
    yeni_s.append(int(fz))

yeni_y=[]
for i in y_test:
    bt["1"]=i[0]
    bt["2"]=i[1]
    bt["3"]=i[2]
    bt["4"]=i[3]
    fz=max(bt,key=bt.get)
    yeni_y.append(int(fz))
    
print("basari_sonucu =>",yeni_s,yeni_y)
plt.plot(yeni_y, label='gerçek')
plt.plot(yeni_s, label='tahmin')
plt.legend()
plt.show()







