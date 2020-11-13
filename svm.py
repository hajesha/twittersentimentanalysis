# Import scikit-learn dataset library
from sklearn.preprocessing import LabelEncoder
from sklearn import datasets
# Import train_test_split function
from sklearn.model_selection import train_test_split
# Import svm model
from sklearn import svm
# Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

if __name__ == '__main__':

    # Create a svm Classifier
    clf = svm.SVC(kernel='linear')  # Linear Kernel

    # read the data
    data = pd.read_csv('./results.csv')

    # print the names of the 13 features
    #print("Features: ", data.feature_names)

    # print the label type of cancer('malignant' 'benign')
    #print("Labels: ", data.target_names)

    # print data(feature)shape
    # data.data.shape

    # print the cancer data features (top 5 records)
    # print(data.data[0:5])

    # print the cancer labels (0:malignant, 1:benign)
    # print(data.target)
    label = data['emotion']
    features = data[['hashtags', 'text', 'emojis']]
    features = features.apply(lambda col: LabelEncoder().fit_transform(
        col.astype(str)), axis=0, result_type='expand')

    msk = np.random.rand(len(features)) < 0.75
    traindata_x = features[msk]
    traindata_y = label[msk].values.tolist()
    testdata_x = features[~msk]
    testdata_y = label[~msk].values.tolist()

    scaling = MinMaxScaler(feature_range=(-1, 1)).fit(traindata_x)
    traindata_x = scaling.transform(traindata_x)
    testdata_x = scaling.transform(testdata_x)

    print("hi1")

    # Train the model using the training sets
    clf.fit(traindata_x, traindata_y)
    print("hi2")

    # Predict the response for test dataset
    y_pred = clf.predict(testdata_x)
    print("hi3")
    # Model Accuracy: how often is the classifier correct?
    print("Accuracy:", metrics.accuracy_score(testdata_y, y_pred))

    # Model Precision: what percentage of positive tuples are labeled as such?
    print("Precision Score : ", precision_score(testdata_y, y_pred, y_pred,
                                                pos_label='positive',
                                                average='micro'))
    print("Recall Score : ", recall_score(testdata_y, y_pred, y_pred,
                                          pos_label='positive',
                                          average='micro'))
