print("What model would you like to use?")
print("1. K Nearest Neighbors")
print("2. Random Forest")
print("3. Decision Tree")
print("")

while True:
    model_choice = input("Enter the number of the model you would like to use: ")
    print("")
    if model_choice in ['1', '2', '3']:
        break
    else:
        print("Invalid input. Please enter 1 for Random Forest, 2 for K Nearest Neighbors, or 3 for Decision Tree.")

print("")

if model_choice == '1':
    from knn import knn
    knn()
elif model_choice == '2':
    from forest import forest
    forest()
else:
    from dectree import tree
    tree()
