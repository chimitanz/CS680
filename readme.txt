To run the training and prediction processes, ensure that the data file is included. These should consist of the image folders and CSV files. The Python script will read these files from the path: ./data/xx, where xx represents the name of the file or folder.

To start the training and prediction processes, run the main.py file. This will initiate the training process and generate the output CSV file.

To skip the training process and use the current model to generate the output CSV file, comment out all instances of the nn.FNN function in main.py and then run the script.