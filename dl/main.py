import os
import matplotlib.pyplot as plt

from ekranui import Ui_MainWindow #(a
from mesajui import Ui_Form #(a

import numpy as np
from PIL import Image
from keras.models import load_model


from subprocess import Popen,PIPE,run

from PyQt5 import QtWidgets,QtGui
from PyQt5.QtCore import QTimer,QProcess
import pyautogui as oto
import sys

def plots2(g1,g2,tg1,tg2,x,y,title,epoch):#(d
    try:
        g11=[i for i in g1]
        g22 =[i for i in g2]
        epochs =[(i+1) for i in range(epoch)]
        boy=len(g1)
        #ekle=0.0
        #if(title=="basari"):
        #    ekle=1.0
        #for i in range(120-boy):
        #    g11.append(ekle)
        #    g22.append(ekle)
        plt.plot(epochs,g11,label=tg1)
        plt.plot(epochs,g22,label=tg2)
        plt.xlabel(x)
        plt.ylabel(y)
        plt.title(title)
        plt.legend()
        #plt.show()
        plt.savefig(title+".png")
        plt.close()
    except Exception as e:
        print(e)


app= QtWidgets.QApplication(sys.argv)


pen = QtWidgets.QMainWindow()
ui = Ui_MainWindow()
ui.setupUi(pen)

msj = QtWidgets.QWidget()
mesaj=Ui_Form()
mesaj.setupUi(msj)

klasor = oto.prompt('Resim Klasörünü Giriniz') #(b
egitim_turu = oto.prompt('Egitim Turunu Grinizi hold / k-fold')
renk_uzayi = oto.confirm(text='Renk Uzayı Seçiniz', title='Renk Uzayı Seçimi', buttons=['rgb', 'hsv','cie'])

print(egitim_turu)
if(klasor==None):
    sys.exit()

val_pro={
"loss_train":[],
"acc_train":[],
"loss_test":[],
"acc_test":[],
"epoch":120,
"tur":0,
"basladi":False,
"p":None,
"tekrar":False,
"egitim_turu":egitim_turu,
"renk_uzayi":renk_uzayi}


def Ana_Program(val):
    print("başı")
    try:
        print(1)
        f=open("kayit.txt","r")
        print(1)
        veri = f.readlines()
        print(1)
        f.close()
        if(len(val["loss_train"])<len(veri)):
            print("yeni veri var")
            val["tur"]+=1;
            print(1)
            son = veri[-1].split(",")
            print(1,son)
            for i in range(len(son)):
                son[i]=float(son[i])
            val["loss_train"].append(son[0])
            print(1)
            val["acc_train"].append(son[1])
            print(1)
            val["loss_test"].append(son[2])
            print(1)
            val["acc_test"].append(son[3])
            print(1)

            ui.label_29.setText("""Eğitim Turu: {}\n
Eğitim Verisi    -> Başarı: {:.4f}, Yitim: {:.4f}\n
Deneme Verisi -> Başarı: {:.4f}, Yitim: {:.4f}""".format(val["tur"],son[1],son[0],son[3],son[2]))

            print(1)
            ui.label_34.setText("Eğitim Yitimi: {:.4f}, Deneme Yitmi: {:.4f}".format(son[0],son[2]))
            print(1)
            ui.label_35.setText("Eğitim Başarı: {:.4f}, Deneme Başarı: {:.4f}".format(son[1],son[3]))
            print(1)
            plots2(val["loss_train"],val["loss_test"],"eğitim yitimi","deneme yitimi","yitim / loss","eğitim Turu","yitim",val["tur"])
            print(1)
            plots2(val["acc_train"],val["acc_test"],"eğitim basarisi","deneme basarisi","basari / acc","eğitim Turu","basari",val["tur"])
            print(1)
            ui.label_30.setPixmap(QtGui.QPixmap("yitim.png"))
            print(1)
            ui.label_33.setPixmap(QtGui.QPixmap("basari.png"))
            print(1)
            if(val["tur"]==120 or len(veri)==120):
                ui.pushButton_2.setText("Eğitimi Tekrar Başlat")
                val["Tekrar"]=True
        else:
            print("yeni veri yok")
    except:
        print("hata var")


deger=False

def egitim_baslat(val):
    print(2,deger)
    if(val["tekrar"]):
        f=open("kayit.txt","w");
        f.close()
        val["tur"]=0
        val["basladi"]==False
    if(val["basladi"]==False):
        print(2)
        son = oto.confirm(text='Eğitimi Başlatmak istediğinize emin misiniz ? ',
                          title='Eğitim', buttons=['Evet', 'Hayir'])
        if(son=="Evet"):
            print(2)
            DEVNULL = open(os.devnull, 'wb')
            here = os.getcwd()+"/cnn_main.py"
            #cmd="python "+here# -d "+str(klasor)
            p = Popen(["python",here,"-d",klasor,"-m",val["egitim_turu"],"-r",val["renk_uzayi"]],shell=False,stdout=DEVNULL, stderr=DEVNULL)
            print("2-oncesi hata")
            val["p"]=p
            #output,outerr = out.communicate()
            print(2)
            val["basladi"]=True
            print(2)
            ui.pushButton_2.setText("Eğitim Suan Devam Ediyor")
            son=[0,0,0,0]
            print(2)
            ui.label_29.setText("""Eğitim Turu: {}\n
Eğitim Verisi    -> Başarı: {}, Yitim: {}\n
Deneme Verisi -> Başarı: {}, Yitim: {}""".format(val["tur"],son[1],son[0],son[3],son[2]))
            print(2)
            ui.label_34.setText("Eğitim Verisi -> Başarı: {}, Yitim: {}".format(son[1],son[0]))
            print(2)
            ui.label_35.setText("Eğitim Verisi -> Başarı: {}, Yitim: {}".format(son[3],son[2]))
    else:
        print("zaten devam etmekte")

baglam = QTimer()
baglam.timeout.connect(lambda : Ana_Program(val_pro))
baglam.start(1000)

def Resim_Sec():
    try:
        resim_yol=QtWidgets.QFileDialog.getOpenFileName()[0]
        ui.pushButton.setText(resim_yol.split("/")[-1])
        ui.label_4.setPixmap(QtGui.QPixmap(resim_yol))

        resim = Image.open(resim_yol)
        resim = resim.resize((200,200))
        resim = np.array(resim)
        model = load_model("cnn_model.h5")
        resim = resim.reshape(1,200,200,3)
        sonuc = model.predict(resim,batch_size=1)[0]
        lst = ["bulutlu","yağmurlu","güneşli","gün batımı"]
        dic = dict(zip(lst, sonuc))
        print(lst,sonuc,dic)
        buyuk = max(dic,key=dic.get)
        ui.label_9.setText(buyuk)
        ui.label_10.setText("bulutlu:{:.4f}\nyağmurlu:{:.4f}\ngüneşli:{:.4f}\ngün batımı:{:.4f}".format(sonuc[0],sonuc[1],sonuc[2],sonuc[3]))
        
        
    except Exception as e:
        print(e)
                        
        
ui.pushButton_2.clicked.connect(lambda :egitim_baslat(val_pro))
ui.pushButton.clicked.connect(Resim_Sec)


pen.show()



sys.exit(app.exec_())
