import ASL_CNN_Fit
import ASL_CNN_Underfit
import ASL_CNN_Overfit

class Main:
    while True:
        while True:
            CNN_Num = input("\nPick a Convolutional Neural Network Architecture:\n\nEnter the number associated:\n\n(1) - Simple \n(2) - Good Fit\n(3) - Overfit\n")
        
            try:
                CNN_Num = int(CNN_Num)
                if CNN_Num not in [1, 2, 3]:
                    raise ValueError
                break
            except ValueError:
                print("Invalid input. Please enter a valid number (1, 2, or 3).")

        while True:
            file = input("\nEnter the name of the file you want to save your model in:\n[DO NOT INCLUDE EXTENSION]\n")

            try:
                with open("./Models/" + file + ".keras", "w") as file:  
                    pass
                break
            except OSError:
                print("Invalid file name or cannot write to file. Please enter a valid file name.")
        
        if CNN_Num == 1:
            CNN = ASL_CNN_Fit.build_test_save(file)
        elif CNN_NUM == 2:
            CNN = ASL_CNN_Underfit.build_test_save(file)
        else:
            CNN = ASL_CNN_Overfit.build_test_save(file)
        
        while True:
            run_again = input("Do you want to run again? (Y/N):\n").strip().upper()
            if run_again in ['Y', 'N']:
                break
            else:
                print("Invalid input. Please enter 'Y' or 'N'.")

        if run_again != 'Y':
            break
                


