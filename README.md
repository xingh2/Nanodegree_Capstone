To run the algorithm, run Image_description.py first. Then open a web page, type in http://localhost:8080. 
On the web page, upload an image and press the button "Submit".

Introduction of the Code:
  1. "Image_description.py" is to run the webpage.
  2. "templates/" includes the html to visualize the website.
  3. "static/" includes all the static files including training code "BOW.py" and "train_and_test.py", all the preprocessing files such as SIFTfeatures, dictionary, and svm, training dataset, test result, and images uploaded by users of the platform.
  4. "static/BOW.py" contains all the functions used to train and test the images including extract SIFT features, construct dictionaries and BOW, train SVM, classify, and calculate the nearest neighbors. 
  5. "static/train_and_test.py" will call the BOW.py to train SVM, test and show the test result.