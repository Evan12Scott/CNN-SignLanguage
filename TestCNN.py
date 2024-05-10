import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image

model = keras.models.load_model("./Models/results.keras")
key = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

im = Image.open("l.png")
im.convert("RGB")

im_arr = np.array(im)
im_arr = np.divide(im_arr, 255)
im_arr = im_arr[:,:,0:1]
im_arr = np.reshape(im_arr, (1, 28, 28))

prediction = model.predict(im_arr)
print(key[np.argmax(prediction)])