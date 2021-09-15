from PIL import Image
import os

data = os.listdir("cloudy")
for resim in data:
    konum = "cloudy/"+resim
    print(konum)
    rsm = Image.open(konum)
    rsm2 = rsm.resize((200,200))
    rsm2.save(konum)
