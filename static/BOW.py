"""
  @author: Xing Hao
  @email: xinghao.1st@gmail.com
  @method: SIFT -> k-means -> SVM
"""
import os
import cv2
import numpy as np
import operator

root_path = os.path.dirname(os.path.realpath(__file__)).replace("\\", "/")
train_path = root_path+"/images"
test_path = root_path+"/images"
svm_dir = root_path+"/svm"
voc_dir = root_path+"/vocabularies"

#Get category name and number of images.
def readFiles(path):
    SetInfo = {}
    print path
    label_names = os.listdir(path)
    for label in label_names:
        folder_path = os.path.join(path, label)
        files_list = os.listdir(folder_path)
        SetInfo[label] = len(files_list)
    return SetInfo

#Calculate features of an image using SIFT.
def SiftFeature(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT(200) # max number of SIFT points is 200
    kp, des = sift.detectAndCompute(gray, None)
    return des

#Initialize list of features for each label.
def initFeatureSet(TrainSetInfo):
    for name, count in TrainSetInfo.items():
        dir = train_path + "/" + name + "/"
        featureSet = np.float32([]).reshape(0, 128)

        print "Extract features from TrainSet " + name + ":"
        print dir
        img_names = os.listdir(dir)
        for img_name in img_names:
            filename = dir + img_name
            img = cv2.imread(filename)
            des = SiftFeature(img)
            featureSet = np.append(featureSet, des, axis=0)

        featCnt = featureSet.shape[0]
        print str(featCnt) + " features in " + str(count) + " images\n"

        # save featureSet to file
        filename = "SIFTFeatures/" + name + ".npy"
        np.save(filename, featureSet)

#Using k-means to calculate the vacabulary (50 words) of each category.
def learnVocabulary(TrainSetInfo):
    wordCnt = 50
    for name, count in TrainSetInfo.items():
        filename = "SIFTFeatures/" + name + ".npy"
        features = np.load(filename)

        print "Learn vocabulary of " + name + "..."
        # use k-means to cluster a bag of features
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.1)
        flags = cv2.KMEANS_RANDOM_CENTERS
        compactness, labels, centers = cv2.kmeans(features, wordCnt, criteria, 20, flags)

        # save vocabulary(a tuple of (labels, centers)) to file
        filename = "vocabularies/" + name + ".npy"
        np.save(filename, (labels, centers))
        print "Done\n"

#Construct bag of words for one image.
def calcFeatVec(features, centers):
    featVec = np.zeros((1, 50))
    for i in range(0, features.shape[0]):
        fi = features[i]
        diffMat = np.tile(fi, (50, 1)) - centers
        sqSum = (diffMat ** 2).sum(axis=1)
        dist = sqSum ** 0.5
        sortedIndices = dist.argsort()
        idx = sortedIndices[0]  # index of the nearest center
        featVec[0][idx] += 1
    return featVec

#Train image classifier using SVM for one category.
def trainClassifier(TrainSetInfo, category_to_train):
    print "================ Training category of " + category_to_train + " ======================="
    trainData = np.float32([]).reshape(0, 50)
    response = np.float32([])
    labels, centers = np.load("vocabularies/" + category_to_train + ".npy")

    for name, count in TrainSetInfo.items():
        print "Init training data of " + name + "..."
        dir = train_path + "/" + name + "/"
        img_names = os.listdir(dir)
        for img_name in img_names:
            filename = dir + img_name
            img = cv2.imread(filename)
            features = SiftFeature(img)
            featVec = calcFeatVec(features, centers)
            trainData = np.append(trainData, featVec, axis=0)

        if name == category_to_train :
            res = np.repeat(np.float32([1]), count)
        else:
            res = np.repeat(np.float32([-1]), count)
        response = np.append(response, res)

    print "Now train svm classifier of " + category_to_train + "..."
    trainData = np.float32(trainData)
    response = response.reshape(-1, 1)
    svm = cv2.SVM()
    svm.train_auto(trainData, response, None, None, None)  # select best params
    svm.save(svm_dir+"/" + category_to_train+".clf")

#Train image classifier using SVM for all categories.
def trainSVM(TrainSetInfo):
    svm_names = os.listdir(svm_dir)
    for name, count in TrainSetInfo.items():
        if name+".clf" in svm_names:
            continue
        trainClassifier(TrainSetInfo, name)

def train():
    TrainSetInfo = readFiles(train_path)
    print "TRAIN: "
    print TrainSetInfo
    #Initialize the features using SIFT. Output: "SIFTFeatures/"
    initFeatureSet(TrainSetInfo)
    # Learn vocabulary of each category using k means. Output: "vocabularies/"
    learnVocabulary(TrainSetInfo)
    # train classifier for each category using SVM. Output: "svm/"
    trainSVM(TrainSetInfo)

#Classify an image, and return the top 5 categories.
def classify(filename, k):
    svm_names = os.listdir(svm_dir)
    svm = cv2.SVM()
    confidence = {}
    for svm_name in svm_names:
        category_name = svm_name[0:-4]
        svm.load(svm_dir+"/"+svm_name)
        labels, centers = np.load(voc_dir+"/" + category_name + ".npy")
        img = cv2.imread(filename)
        features = SiftFeature(img)
        featVec = calcFeatVec(features, centers)
        case = np.float32(featVec)
        confidence[category_name] = 2*(svm.predict(case, False) - 0.5)*svm.predict(case, True)
    sorted_confidence = sorted(confidence.items(), key=operator.itemgetter(1), reverse=True)
    result = []
    for tuple in sorted_confidence[0:k]:
        result.append(tuple[0])
    return result

#Test the accuracy of classifier.
def test(path=test_path):
    TestSetInfo = readFiles(path)
    print TestSetInfo
    total_img = 0
    total_correct = 0
    result_names = os.listdir(root_path+"/test_result")
    for name, count in TestSetInfo.items():
        if name+".txt" in result_names:
            continue
        total = 0
        correct = 0
        dir = path + "/" + name + "/"
        predictions = ""
        print "Classify on TestSet '" + name + "':"
        img_names = os.listdir(dir)
        for img_name in img_names:
            total += 1
            filename = dir + img_name
            predict_name = classify(filename)
            predictions += img_name+"\t"+', '.join(predict_name)+"\r\n"
            if name in predict_name:
                correct += 1

        print "Accuracy of "+name+": " + str(correct) + " / " + str(total) + "\n"
        file_object = open(root_path+"/test_result/"+name+".txt", 'w')
        file_object.write(predictions+"Accuracy of "+name+": " + str(correct) + " / " + str(total) + "\n")
        file_object.close()
        total_img += total
        total_correct += correct

    print "Total accuracy: " + str(total_correct) + " / " + str(total_img)

# find k nearest neighbors in categories for an image.
def knn(image, categories, k=10):
    dictionary = {}
    for category in categories:
        labels, centers = np.load(voc_dir + "/" + category + ".npy")
        img = cv2.imread(image)
        features = SiftFeature(img)
        featVec = calcFeatVec(features, centers)
        target_case = np.float32(featVec)
        path = train_path+"/"+category
        names = os.listdir(path)
        for name in names:
            filename = path+"/"+name
            img = cv2.imread(filename)
            features = SiftFeature(img)
            featVec = calcFeatVec(features, centers)
            case = np.float32(featVec)
            dictionary[category+"/"+name] = np.linalg.norm(case-target_case)
    list = []
    i=0
    for w in sorted(dictionary.items(), key=operator.itemgetter(1), reverse=False):
        list.append(w[0])
    return list[0:k]