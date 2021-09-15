from PIL import Image
import os

data = os.listdir("rainy")
for resim in data:
    konum = "rainy/"+resim
    print(konum)
    rsm = Image.open(konum)
    rsm2 = rsm.resize((200,200))
    rsm2.save(konum)
