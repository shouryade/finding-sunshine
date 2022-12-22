import cv2
import os
import pickle

from utils import get_encodings

SUN_PATH = os.path.join("faces", "sunshine")

count = 1

# Creating empty dictionary to store encodings, currently sunshine ONLY
encodings_dict = {}


for image in os.listdir(SUN_PATH):
    impath = os.path.join("faces", "sunshine", image)
    image = cv2.imread(impath)
    encodings = get_encodings(image)
    name = "sunshine"

    e = encodings_dict.get(name, [])
    e.extend(encodings)
    encodings_dict[name] = e
    count += 1

# print(encodings_dict)

with open("encodings", "wb") as f:
    pickle.dump(encodings_dict, f)
