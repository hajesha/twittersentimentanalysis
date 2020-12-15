import pandas as pd
import re
import nltk
import preprocessor as p    #If this doesnt work then please manually import " tweet-preprocessor "
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from ekphrasis.classes.segmenter import Segmenter
from sklearn.model_selection import train_test_split

nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('wordnet')


def readfile(filepath):
    return pd.read_csv(filepath, encoding='utf-8', memory_map=True)


def cleanupText(words):
    new_words = []
    stop_words = set(stopwords.words('english'))
    for word in words:
        if word not in stop_words:
            withoutlink = re.sub(' ', '-----', word)
            cleanWord = p.clean(withoutlink)
            new_word = re.sub('[^a-z\s]', '', cleanWord)
            if new_word != '' and new_word != "-----" and new_word != "rt":
                new_words.append(new_word)
    return new_words


def splitUpTweets(data, corpus):
    if data != []:
        listToStr1 = ' '.join([str(elem.lower()) for elem in data])
        return corpus.segment(listToStr1).split()


def extractHashtags(dataset):
    seg_tw = Segmenter(corpus="twitter")
    stop_words = set(stopwords.words('english'))
    dataset['hashtags'] = dataset['text'].apply(lambda x: re.findall(r"#(\w+)", x)).apply(lambda x: splitUpTweets(x, seg_tw))

    # Remove stop words in segmented tweet
    for i in range(len(dataset['hashtags'])):
        if dataset['hashtags'][i] is not None:
            dataset['hashtags'][i] = list(filter(lambda a: ((a not in stop_words) & (a != "_")), dataset['hashtags'][i]))
    return dataset


def tokenizeAndLem(dataset):
    # Lemmatization and tokenization
    lemmatizer = nltk.stem.WordNetLemmatizer()
    tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True)

    dataset['text'] = dataset['text'].apply(
        lambda x: [(lemmatizer.lemmatize(w)) for w in tokenizer.tokenize(x)])

    return dataset

def cleanUpText(name, newname, extractHashtag = False, tokenAndLem = True, test_split = 0.2, createValSet = True):
    # Initialize all the dataframes
    df_pd = readfile(name + ".csv")

    # Extracting and parsing the hashtags
    if extractHashtag:
        df_pd = extractHashtags(df_pd)

    if tokenAndLem:
        df_pd = tokenizeAndLem(df_pd)
        df_pd = df_pd.dropna(subset=['text'])

    # CLean up the text
    df_pd['text'] = df_pd['text'].apply(cleanupText)

    df_pd = df_pd.dropna(subset=['text'])
    df_pd.reset_index(drop=True, inplace=True)

    X_train, X_testval, y_train, y_testval = train_test_split(df_pd.text, df_pd.emotion, test_size=test_split, random_state=123)
    trainingset = pd.DataFrame({'text': X_train.values, 'emotion': y_train.values}, columns=["text", "emotion"])
    trainingset.to_csv(newname + 'training.csv', encoding='utf-8')

    if createValSet:
        testandval = pd.DataFrame({'text': X_testval.values, 'emotion': y_testval.values}, columns=["text", "emotion"])
        X_test, X_val, y_test, y_val = train_test_split(testandval.text, testandval.emotion, test_size=0.5, random_state=123)
    
        testSet = pd.DataFrame({'text': X_test.values, 'emotion': y_test.values}, columns=["text", "emotion"])
        testSet.to_csv(newname + 'test.csv', encoding='utf-8')
    
        valSet = pd.DataFrame({'text': X_val.values, 'emotion': y_val.values}, columns=["text", "emotion"])
        valSet.to_csv(newname + 'val.csv', encoding='utf-8')
    else:
        testSet = pd.DataFrame({'text': X_testval.values, 'emotion': y_testval.values}, columns=["text", "emotion"])
        testSet.to_csv(newname + 'test.csv', encoding='utf-8')

if __name__ == '__main__':
    name = "balancedDataNeutral"
    newname = "balancedDatasetNeutral"
    cleanUpText(name, newname)
