"""
  @author: Xing Hao
  @email: xinghao.1st@gmail.com
"""

import BOW
import os

def accuracy():
    result_dir = "D:/Dropbox/Xing/Capstone/static/test_result"
    files_list = os.listdir(result_dir)
    for file in files_list:
        with open(result_dir + "/" + file) as f:
            lines = f.readlines()
        print lines[-1].strip()

def accuracy(k):
    result_dir = "D:/Dropbox/Xing/Capstone/static/test_result"
    files_list = os.listdir(result_dir)
    for file in files_list:
        correct = 0
        total = 0
        items = file.split(".")
        with open(result_dir + "/" + file) as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            names = line.split("\t")
            if len(names)<2:
                continue
            total += 1
            categories = names[1].split(", ")
            if items[0] in categories[:k] :
                correct+=1
        print items[0],"\t",correct,"\t",total



if __name__ == "__main__":
    # Train (Takes very long time)
    #BOW.train()

    # Test (Takes very long time)
    #BOW.test("D:/Dropbox/Xing/Capstone/static/images")
    # Show Test Result
    accuracy(4)


    # Classify one image
    #print BOW.classify("D:/Dropbox/Xing/Capstone/static/images/accordion/image_0001.jpg", 5)


