from ASL_CNN_Balanced import Balanced_Model
from ASL_CNN_Simple import Simple_Model
from ASL_CNN_Complex import Complex_Model

class Main:
    # execute program till user stops
    while True:
        # model type handler
        while True:
            CNN_Num = input("\nPick a Convolutional Neural Network Architecture:\n\nEnter the number associated:\n\n(1) - Simple \n(2) - Balanced\n(3) - Complex\n")
        
            try:
                CNN_Num = int(CNN_Num)
                if CNN_Num not in [1, 2, 3]:
                    raise ValueError
                break
            except ValueError:
                print("Invalid input. Please enter a valid number (1, 2, or 3).")
        # file input handler
        while True:
            file_input = input("\nEnter the name of the file you want to save your model in:\n[DO NOT INCLUDE EXTENSION]\n")

            try:
                with open("./Models/" + file_input + ".keras", "w") as f:
                    pass
                break
            except OSError:
                print("Invalid file name or cannot write to file. Please enter a valid file name.")
        file = "./Models/" + file_input + ".keras"

        # load and test appropriate model based on user input
        if CNN_Num == 1:
            CNN = Simple_Model.build_test_save(file)
        elif CNN_Num == 2:
            CNN = Balanced_Model.build_test_save(file)
        else:
            CNN = Complex_Model.build_test_save(file)
        # continue program execution handler
        while True:
            run_again = input("Do you want to run again? (Y/N):\n").strip().upper()
            if run_again in ['Y', 'N']:
                break
            else:
                print("Invalid input. Please enter 'Y' or 'N'.")

        if run_again != 'Y':
            break
                


