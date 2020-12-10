import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import BernoulliNB
from sklearn import svm

seed = 123

def assess(X, y, models, cv=5, scoring=['precision', 'recall', 'f1', 'roc_auc']):
    results = pd.DataFrame()
    for name, model in models:
        result = pd.DataFrame(cross_validate(model, X, y, cv=cv,scoring=scoring))
        mean = result.mean().rename('{}_mean'.format)
        results[name] = pd.concat([mean], axis=0)
    return results.sort_index()

def create_baseline_models():
    """Create list of baseline models."""
    models = []
    models.append(('log', LogisticRegression(random_state=seed)))
    models.append(('sgd', SGDClassifier(random_state=seed)))
    models.append(('nb', BernoulliNB()))
    models.append(('svm', svm.SVC()))
    models.append(('abc',  AdaBoostClassifier()))
    models.append(('rfc', RandomForestClassifier( random_state=0)))
    return models

if __name__ == '__main__':

    models = create_baseline_models()
    data = pd.read_csv('./balancedCombinedProcessedNoValtraining.csv')
    test = pd.read_csv('./balancedCombinedProcessedNoValtest.csv')
    features = data.text
    labels = data.emotion
    featurestest = test[:200].text
    labelstest = test[:200].emotion

    tfidfconverter = TfidfVectorizer()
    features = tfidfconverter.fit_transform(features).toarray()
    featurestest = tfidfconverter.transform(featurestest).toarray()
    print(assess(features, labels, models))