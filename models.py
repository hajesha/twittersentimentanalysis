import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.tree import DecisionTreeClassifier


def RF_model(traindata_x, traindata_y, testdata_x):

    # Create a svm Classifier
    clf = RandomForestClassifier(criterion='gini', max_depth=274,
                           max_features='log2', min_samples_leaf=12, min_samples_split=5, n_estimators=267)

    # Train the model using the training sets
    clf.fit(traindata_x, traindata_y)

    # Predict the response for test dataset
    return clf.predict(testdata_x)


def NaiveBayes_model(traindata_x, traindata_y, testdata_x):
    # Create a NB Classifier
    clf = MultinomialNB()

    # Train the model using the training sets
    clf.fit(traindata_x, traindata_y)

    # Predict the response for test dataset
    return clf.predict(testdata_x)


def LogisticRegression_model(traindata_x, traindata_y, testdata_x):
    # Create a Logistic Regression Classifier
    clf = LogisticRegression()

    # Train the model using the training sets
    clf.fit(traindata_x, traindata_y)

    # Predict the response for test dataset
    return clf.predict(testdata_x)


def Ada_model(traindata_x, traindata_y, testdata_x):
    # Create a Logistic Regression Classifier
    clf = AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=1),
        n_estimators=200
    )

    # Train the model using the training sets
    clf.fit(traindata_x, traindata_y)

    # Predict the response for test dataset
    return clf.predict(testdata_x)


def printMetrics(predictions, trueLabels, name):
    print(accuracy_score(trueLabels, predictions))
    # print(confusion_matrix(trueLabels, predictions))
    print(classification_report(trueLabels, predictions))

if __name__ == '__main__':


# Final Results -----------------------------------------------------------------------------------
    data = pd.read_csv('balancedDatasettraining.csv')
    testing = pd.read_csv('balancedDatasettest.csv')
    print("Final Results")
    traindata_y = data.emotion
    traindata_x = data.text

    testdata_y = testing.emotion
    testdata_x = testing.text

    converter = TfidfVectorizer()
    traindata_x = converter.fit_transform(traindata_x)
    testdata_x = converter.transform(testdata_x)

    # RF
    svmprediction = RF_model(traindata_x, traindata_y, testdata_x)
    printMetrics(svmprediction, testdata_y, "RF")

    # NB
    svmprediction = NaiveBayes_model(traindata_x, traindata_y, testdata_x)
    printMetrics(svmprediction, testdata_y, "MNB")

    # LR
    svmprediction = LogisticRegression_model(traindata_x, traindata_y, testdata_x)
    printMetrics(svmprediction, testdata_y, "LR")

    # Ada
    svmprediction = Ada_model(traindata_x, traindata_y, testdata_x)
    printMetrics(svmprediction, testdata_y, "ADA")
# No Processing -----------------------------------------------------------------------------------
    print("No Processing")
    data = pd.read_csv('balancedDatasetNoProcesstraining.csv')
    testing = pd.read_csv('balancedDatasetNoProcesstest.csv')

    traindata_y = data.emotion
    traindata_x = data.text

    testdata_y = testing.emotion
    testdata_x = testing.text

    converter = TfidfVectorizer()
    traindata_x = converter.fit_transform(traindata_x)
    testdata_x = converter.transform(testdata_x)

    # RF
    svmprediction = RF_model(traindata_x, traindata_y, testdata_x)
    printMetrics(svmprediction, testdata_y, "RF")

    # NB
    svmprediction = NaiveBayes_model(traindata_x, traindata_y, testdata_x)
    printMetrics(svmprediction, testdata_y, "MNB")

    # LR
    svmprediction = LogisticRegression_model(traindata_x, traindata_y, testdata_x)
    printMetrics(svmprediction, testdata_y, "LR")

    # Ada
    svmprediction = Ada_model(traindata_x, traindata_y, testdata_x)
    printMetrics(svmprediction, testdata_y, "ADA")
# Neutral class -----------------------------------------------------------------------------------
    print("Neutral class")
    data = pd.read_csv('balancedDatasetNeutraltraining.csv')
    testing = pd.read_csv('balancedDatasetNeutraltest.csv')

    traindata_y = data.emotion
    traindata_x = data.text

    testdata_y = testing.emotion
    testdata_x = testing.text

    converter = TfidfVectorizer()
    traindata_x = converter.fit_transform(traindata_x)
    testdata_x = converter.transform(testdata_x)

    # RF
    svmprediction = RF_model(traindata_x, traindata_y, testdata_x)
    printMetrics(svmprediction, testdata_y, "RF")

    # NB
    svmprediction = NaiveBayes_model(traindata_x, traindata_y, testdata_x)
    printMetrics(svmprediction, testdata_y, "MNB")

    # LR
    svmprediction = LogisticRegression_model(traindata_x, traindata_y, testdata_x)
    printMetrics(svmprediction, testdata_y, "LR")

    # Ada
    svmprediction = Ada_model(traindata_x, traindata_y, testdata_x)
    printMetrics(svmprediction, testdata_y, "ADA")
# Hashtag class -----------------------------------------------------------------------------------
    data = pd.read_csv('balancedDatasetHashtagtraining.csv')
    testing = pd.read_csv('balancedDatasetHashtagtest.csv')

    print("Multiple")
    traindata_y = data.emotion
    traindata_x = data[['text', 'hashtags']]

    testdata_y = testing.emotion
    testdata_x = testing[['text', 'hashtags']]

    converter = LabelEncoder()
    traindata_x = traindata_x.apply(lambda col: converter.fit_transform(
        col.astype(str)), axis=0, result_type='expand')

    testdata_x = testdata_x.apply(lambda col: converter.fit_transform(
        col.astype(str)), axis=0, result_type='expand')

    # RF
    svmprediction = RF_model(traindata_x, traindata_y, testdata_x)
    printMetrics(svmprediction, testdata_y, "RF")

    # NB
    svmprediction = NaiveBayes_model(traindata_x, traindata_y, testdata_x)
    printMetrics(svmprediction, testdata_y, "MNB")

    # LR
    svmprediction = LogisticRegression_model(traindata_x, traindata_y, testdata_x)
    printMetrics(svmprediction, testdata_y, "LR")

    # Ada
    svmprediction = Ada_model(traindata_x, traindata_y, testdata_x)
    printMetrics(svmprediction, testdata_y, "ADA")

# Unbalanced class -----------------------------------------------------------------------------------
    data = pd.read_csv('balancedDatasetHashtagtraining.csv')
    testing = pd.read_csv('balancedDatasetHashtagtest.csv')

    print("Unbalanced")
    traindata_y = data.emotion
    traindata_x = data[['text', 'hashtags']]

    testdata_y = testing.emotion
    testdata_x = testing[['text', 'hashtags']]

    converter = LabelEncoder()
    traindata_x = traindata_x.apply(lambda col: converter.fit_transform(
        col.astype(str)), axis=0, result_type='expand')

    testdata_x = testdata_x.apply(lambda col: converter.fit_transform(
        col.astype(str)), axis=0, result_type='expand')

    # RF
    svmprediction = RF_model(traindata_x, traindata_y, testdata_x)
    printMetrics(svmprediction, testdata_y, "RF")

    # NB
    svmprediction = NaiveBayes_model(traindata_x, traindata_y, testdata_x)
    printMetrics(svmprediction, testdata_y, "MNB")

    # LR
    svmprediction = LogisticRegression_model(traindata_x, traindata_y, testdata_x)
    printMetrics(svmprediction, testdata_y, "LR")

    # Ada
    svmprediction = Ada_model(traindata_x, traindata_y, testdata_x)
    printMetrics(svmprediction, testdata_y, "ADA")