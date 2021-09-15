import cv2
from PIL import Image
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# A 

data = os.listdir('dataset2');
#print(data)

rgb=[]
hsv=[]
cie=[]
hata=[]
print("bilgi: resimler grb formatta okunuyor...\n")
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
print("\nbilgi: resimler rgb formatta surf,sırf öznitelikleri çıkartılıyor\n")


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
    print(i,"/",boy," kadar resimin öz niteliği çıkartıldı")
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
    return don # O resimi 24 parcaya bolduk ve her parcada ne kadar yer işaretlenmiş baktık


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
print(veri_kumesi.head,"\n")

print("\bilgi: Makine Öğrenimi algoritmaları başlatılıyor\n")
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.metrics import auc,r2_score,mean_absolute_error,mean_squared_error,median_absolute_error,mean_absolute_error,roc_curve

print("\bilgi: Veri Seti Ayrıştırılıyor \n")
veri_x= veri_kumesi.drop(["dosya","y"],axis=1)
veri_y = veri_kumesi["y"]
x_train, x_test, y_train, y_test = train_test_split(veri_x,veri_y,test_size=0.2,random_state=10)


regrasyon = LogisticRegression()

print("\bilgi: Eğitim Gerçekleştiriliyor \n")
x_train = x_train.astype('int')
x_test = x_test.astype('int')
y_train = y_train.astype('int')
y_test = y_test.astype('int')

regrasyon.fit(x_train,y_train)
y_predicted = regrasyon.predict(x_test) 

ortlama_kare_hata = mean_squared_error(y_test, y_predicted, squared=False)
basari = r2_score(y_test, y_predicted)
ortlama_mutlak_hata = mean_absolute_error(y_test, y_predicted)
medyan_hata = median_absolute_error(y_test, y_predicted)

print("ortlama_mutlak_hata=>",ortlama_mutlak_hata,
      "ortlama_kare_hata =>",ortlama_kare_hata,
      "medyan_hata=>",medyan_hata,
      "basarılı =>",basari)

"""
y_test_lst=list(y_test)
for i in range(1,5):
    y_t=[]
    y_p=[]
    for j in range(len(y_test_lst)):
        if(int(y_test_lst[j])==i):
            qr=int(j/i)
            y_t.append(qr)
            pe=y_predicted[j]/i
            y_p.append(pe)
    fpr,tpr,th= roc_curve(y_t,y_p,pos_label=[True,False])
    roc_auc=auc(fpr,tpr)
    plt.title("ROC Eğrisi")
    plt.plot(fpr,tpr,'b',label="AUC=%0.2f"%roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],"r--")
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.ylabel("TP sayısı")
    plt.xlabel("FP sayisi")
    plt.save("roc_mo"+str(i)+".png")
    plt.close()
"""

plt.plot(y_test, label='gerçek')
plt.plot(y_predicted, label='tahmin')
plt.legend()
plt.show()


