import matplotlib.pyplot as plt

# Import scikit-learn dataset library
from sklearn.preprocessing import LabelEncoder
from sklearn import datasets
# Import train_test_split function
from sklearn.model_selection import train_test_split
# Import svm model
from sklearn import svm
# Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from tpot import TPOTClassifier


if __name__ == '__main__':

    data = pd.read_csv('./results.csv')

    # Create a svm Classifier
    clf = svm.SVC(kernel='rbf',  gamma='auto', C=0.01, degree=3)

    label = data['emotion']
    features = data[['text', 'hashtags', 'emojis',
                     'exaggerate_punctuation', 'pos_tag']]

    features = features.apply(lambda col: LabelEncoder().fit_transform(
        col.astype(str)), axis=0, result_type='expand')

    msk = np.random.rand(len(features)) < 0.7
    traindata_x = features[msk]
    traindata_y = label[msk].values.tolist()
    testdata_x = features[~msk]
    testdata_y = label[~msk].values.tolist()

    scaling = MinMaxScaler(feature_range=(-1, 1)).fit(traindata_x)
    traindata_x = scaling.transform(traindata_x)
    testdata_x = scaling.transform(testdata_x)

    # Train the model using the training sets
    clf.fit(traindata_x, traindata_y)

    # Predict the response for test dataset
    y_pred = clf.predict(testdata_x)
    # Model Accuracy: how often is the classifier correct?
    print("Accuracy:", metrics.accuracy_score(testdata_y, y_pred))

    # Model Precision: what percentage of positive tuples are labeled as such?
    print("Precision Score : ", metrics.precision_score(testdata_y, y_pred,
                                                        pos_label='positive',
                                                        average='weighted', zero_division=0))
    print("Recall Score : ", metrics.recall_score(testdata_y, y_pred,
                                                  pos_label='positive',
                                                  average='weighted'))

    print(metrics.f1_score(testdata_y, y_pred, average='macro'))

    print(metrics.f1_score(testdata_y, y_pred, average='micro'))

    print(metrics.f1_score(testdata_y, y_pred, average='weighted'))
