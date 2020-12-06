import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB, ComplementNB, CategoricalNB
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics


# REFERENCES FOR ME
# https://stackoverflow.com/questions/3473612/ways-to-improve-the-accuracy-of-a-naive-bayes-classifier

if __name__ == '__main__':
    # read the data
    data = pd.read_csv('./results.csv')
    # features = data.drop(['original_text','emotion', 'exaggerate_punctuation'], axis=1)
    features = data[['text', 'hashtags', 'emojis',
                     'exaggerate_punctuation', 'pos_tag']]
    # features = data[['text']]
    features = features.apply(lambda col: LabelEncoder().fit_transform(
        col.astype(str)), axis=0, result_type='expand')
    label = data['emotion']

    # split the data into 75% training and 25% testing
    msk = np.random.rand(len(features)) < 0.70
    traindata_x = features[msk]
    traindata_y = label[msk].values.tolist()
    testdata_x = features[~msk]
    testdata_y = label[~msk].values.tolist()

    classifier = BernoulliNB()

    # fit the model
    classifier.fit(traindata_x, traindata_y)

    # predict
    pred = classifier.predict(testdata_x)

    print("Accuracy:", metrics.accuracy_score(testdata_y, pred))
    print("\nClassification Report:",
          metrics.classification_report(testdata_y, pred, zero_division=0))

    clf_4 = AdaBoostClassifier()
    clf_4.fit(traindata_x, traindata_y)
    pred2 = clf_4.predict(testdata_x)

    print("Accuracy:", metrics.accuracy_score(testdata_y, pred2))
    print("\nClassification Report:",
          metrics.classification_report(testdata_y, pred2, zero_division=0))

    clf_4 = RandomForestClassifier()
    clf_4.fit(traindata_x, traindata_y)
    pred2 = clf_4.predict(testdata_x)

    print("Accuracy:", metrics.accuracy_score(testdata_y, pred2))
    print("\nClassification Report:",
          metrics.classification_report(testdata_y, pred2, zero_division=0))
# try k-fold tfidf bow
