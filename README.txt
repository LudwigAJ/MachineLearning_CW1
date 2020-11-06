# MachineLearning_CW1

A Machine Learning model employing Decision Trees in order to be able to correctly predict which room an individual is in based on the Wi-Fi signal strength that person's device receives.

A program that takes in a .txt file containing Wi-Fi strengths from different routers recorded by an agent in a room.
.txt file must contain data formated as the Wi-Fi strength given in each column corresponding to their router, with the room being labeled in the last column.

Inside the CO395-CW.py there is a function named main. Inside this function we call runTest() which takes three parameters. 
The first one specifies the path to the training dataset, the 2nd parameter specifies the number of folds, and the last parameter takes in a boolean specifying if we should print the tree while its created or not. 
We did not include any command-line arguments.

Edit the CO395-CW.py file to specify dataset, number of folds, and True to show tree or False to hide.
Pruning of trees is on by default. Although, a modification of our file could disable it.

Run the program by running the following command in the root directory of this repository:
  python3 CO395-CW.py
  

Part of coursework for COMP97101(CO395) Machine Learning at Imperial College London
