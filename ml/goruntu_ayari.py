from PIL import Image
import os

data = os.listdir("dataset2")
for resim in data:
    try:
        konum = "dataset2/"+resim
        print(konum)
        rsm = Image.open(konum)
        rsm2 = rsm.resize((600,400))
        rsm2.save(konum)
    except:
        print("sil=>",konum)
