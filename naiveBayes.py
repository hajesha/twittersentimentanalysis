import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB, ComplementNB, CategoricalNB
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.model_selection import train_test_split


# REFERENCES FOR ME
# https://stackoverflow.com/questions/3473612/ways-to-improve-the-accuracy-of-a-naive-bayes-classifier

if __name__ == '__main__':
    # read the data
    data = pd.read_csv('./resultsHK.csv')
    features = data.text
    labels = data.emotion


    vectorizer = TfidfVectorizer(max_features=2500, min_df=7, max_df=0.8)
    processed_features = vectorizer.fit_transform(features).toarray()
    X_train, X_test, y_train, y_test = train_test_split(processed_features, labels, test_size=0.2, random_state=0)

    classifier = BernoulliNB()
    classifier.fit(X_train, y_train)

    # predict
    pred = classifier.predict(X_test)

    print("Accuracy:", metrics.accuracy_score(y_test, pred))
    print("\nClassification Report:", metrics.classification_report(y_test, pred, zero_division=0))


    # data = data.sample(frac=1).reset_index(drop=True)
    # # features = data.drop(['original_text','emotion', 'exaggerate_punctuation'], axis=1)
    # features = data[['text']]
    # # features = data[['text']]
    # features = features.apply(lambda col: LabelEncoder().fit_transform(
    #     col.astype(str)), axis=0, result_type='expand')
    # label = data['emotion']
    #
    # # split the data into 75% training and 25% testing
    # msk = np.random.rand(len(features)) < 0.75
    # traindata_x = features[msk]
    # traindata_y = label[msk].values.tolist()
    # testdata_x = features[~msk]
    # testdata_y = label[~msk].values.tolist()
    #
    # classifier = BernoulliNB()
    #
    # # fit the model
    # classifier.fit(traindata_x, traindata_y)
    #
    # # predict
    # pred = classifier.predict(testdata_x)
    #

    #
    #
    # # clf_4 = AdaBoostClassifier()
    # # clf_4.fit(traindata_x, traindata_y)
    # # pred2 = clf_4.predict(testdata_x)
    # #
    # # print("Accuracy:", metrics.accuracy_score(testdata_y, pred2))
    # # print("\nClassification Report:",
    # #       metrics.classification_report(testdata_y, pred2, zero_division=0))
    # #
    # # clf_4 = RandomForestClassifier()
    # # clf_4.fit(traindata_x, traindata_y)
    # # pred2 = clf_4.predict(testdata_x)
    # #
    # # print("Accuracy:", metrics.accuracy_score(testdata_y, pred2))
    # # print("\nClassification Report:",
    # #       metrics.classification_report(testdata_y, pred2, zero_division=0))
