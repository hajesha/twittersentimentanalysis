import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics

if __name__ == '__main__':
    # read the data
    data = pd.read_csv('./results.csv')
    # features = data.drop(['original_text','emotion', 'exaggerate_punctuation'], axis=1)
    features = data[['text', 'hashtags', 'emojis',
                     'exaggerate_punctuation', 'pos_tag']]
    features = features.apply(lambda col: LabelEncoder().fit_transform(
        col.astype(str)), axis=0, result_type='expand')
    # TODO fix this
    label = data['emotion']
    # label = label.apply(lambda col: LabelEncoder().fit_transform(col.astype(str)), axis=0, result_type='expand')

    # split the data into 75% training and 25% testing
    msk = np.random.rand(len(features)) < 0.75
    traindata_x = features[msk]
    traindata_y = label[msk].values.tolist()
    testdata_x = features[~msk]
    testdata_y = label[~msk].values.tolist()

    classifier = MultinomialNB()

    # fit the model
    classifier.fit(traindata_x, traindata_y)

    # predict
    pred = classifier.predict(testdata_x)

    print("Accuracy:", metrics.accuracy_score(testdata_y, pred))
    print("\nClassification Report:",
          metrics.classification_report(testdata_y, pred, zero_division=0))
