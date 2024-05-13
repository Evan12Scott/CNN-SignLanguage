"""
Authors: Evan Scott, Kieran Kennedy, Sean Pala
Last Modified: 5/12/24
Description:
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image

def main():

    while True:
        CNN_Model = input("Please select a trained model from the Models directory to use: ")
        
        try:
            if ".keras" not in CNN_Model:
                raise FileNotFoundError
            open("./Models/" + CNN_Model)
            break
        except FileNotFoundError:
            print("Invalid file name. Please enter a valid file.")

    model = keras.models.load_model("./Models/" + CNN_Model)
    key = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

    total_correct = 0
    incorrect_letter = []
    for letter in key:
        if letter not in ["J", "Z"]:
            im = Image.open("./Images/" + letter + ".png")
            im.convert("RGB")

            im_arr = np.array(im)
            im_arr = np.divide(im_arr, 255)
            im_arr = im_arr[:,:,0:1]
            im_arr = np.reshape(im_arr, (1, 28, 28))

            prediction = model.predict(im_arr)
            if key[np.argmax(prediction)] == letter:
                total_correct += 1
            else:
                incorrect_letter.append(letter)
    
    print("Testing complete,", (str(total_correct) + "/24"), ("("+ str((total_correct/24)*100) + "%)"))
    print("Incorrectly identified: ", incorrect_letter)
    

if __name__ == "__main__":
    main()