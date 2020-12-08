import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from ast import literal_eval

if __name__ == '__main__':
    # read the data
    data = pd.read_csv('./resultsPOG.csv')
    features = data.text
    labels = data.emotion

    tfidfconverter = TfidfVectorizer()
    X = tfidfconverter.fit_transform(features).toarray()

    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=0)
    #
    # TF IDF using random forest classifier
    text_classifier = RandomForestClassifier(n_estimators=100, random_state=0)
    text_classifier.fit(X_train, y_train)

    predictions = text_classifier.predict(X_test)
    print(accuracy_score(y_test, predictions))
    print(confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions))

    # # # TF IDF using random Naive Bayes Bernolli
    # text_classifier = BernoulliNB()
    # text_classifier.fit(X_train, y_train)
    #
    # predictions = text_classifier.predict(X_test)
    # print(accuracy_score(y_test, predictions))
    # print(confusion_matrix(y_test, predictions))
    # print(classification_report(y_test, predictions))
    #
    # Regular Naive Bayes
    # le = preprocessing.MultiLabelBinarizer()
    # features = le.fit_transform(features.apply(literal_eval))
    # features = np.apply_along_axis(le.fit_transform, 0, features)
    # X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=0)
    # text_classifier = BernoulliNB()
    # text_classifier.fit(X_train, y_train)
    #
    # predictions = text_classifier.predict(X_test)
    # print(confusion_matrix(y_test, predictions))
    # print(classification_report(y_test, predictions))
    # print(accuracy_score(y_test, predictions))

    # # #Adaboost
    # classifier = AdaBoostClassifier()
    # classifier.fit(X_train, y_train)
    # pred = classifier.predict(X_test)
    # print(confusion_matrix(y_test, pred))
    # print(classification_report(y_test, pred))
    # print(accuracy_score(y_test, pred))
    #
    # text_classifier = RandomForestClassifier(n_estimators=100, random_state=0)
    # text_classifier.fit(X_train, y_train)
    #
    # predictions = text_classifier.predict(X_test)
    # print(accuracy_score(y_test, predictions))
    # print(confusion_matrix(y_test, predictions))
    # print(classification_report(y_test, predictions))
