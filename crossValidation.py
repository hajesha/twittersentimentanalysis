import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.preprocessing import MultiLabelBinarizer

seed = 123

def assess(X, y, models, cv=5, scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc']):
    results = pd.DataFrame()
    for name, model in models:
        result = pd.DataFrame(cross_validate(model, X, y, cv=cv, scoring=scoring))
        mean = result.mean().rename('{}_mean'.format)
        results[name] = pd.concat([mean], axis=0)
    return results.sort_index()


def create_baseline_models():
    models = []
    models.append(('log', LogisticRegression(random_state=seed)))
    models.append(('sgd', SGDClassifier(random_state=seed)))
    models.append(('nb', MultinomialNB()))
    models.append(('svm', svm.SVC()))
    models.append(('abc',  AdaBoostClassifier(base_estimator=MultinomialNB())))
    models.append(('rfc', RandomForestClassifier(random_state=0)))
    return models

def encodeTheStuff(models, features, labels, encodeLabel="TFIDF"):
    if encodeLabel == "TFIDF" or encodeLabel == "All":
        print("TF IDF")
        converter = TfidfVectorizer()
        features1 = converter.fit_transform(features)
        print(assess(features1, labels, models))

    if encodeLabel == "OH" or encodeLabel == "All":
        print("\nOne Hot")
        converter = MultiLabelBinarizer()
        features2 = converter.fit_transform(features)
        print(assess(features2, labels, models))

    if encodeLabel == "BoW" or encodeLabel == "All":
        print("\nBags of Words")
        converter = CountVectorizer()
        features3 = converter.fit_transform(features)
        print(assess(features3, labels, models))


if __name__ == '__main__':
    models = create_baseline_models()
    data = pd.read_csv('./balancedDatasettest.csv')
    features = data.text
    labels = data.emotion
    encodeTheStuff(models, features, labels, "All")
