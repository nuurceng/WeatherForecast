from PIL import Image
import os

data = os.listdir("shine")
for resim in data:
    konum = "shine/"+resim
    print(konum)
    rsm = Image.open(konum)
    rsm2 = rsm.resize((200,200))
    rsm2.save(konum)
