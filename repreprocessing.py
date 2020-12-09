import pandas as pd
import re
import nltk
import preprocessor as p    #If this doesnt work then please manually import " tweet-preprocessor "
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from ekphrasis.classes.segmenter import Segmenter
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

def cleanUpText(name, newname, extractHashtag = False, tokenAndLem = True):
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

    df_pd[['text','emotion']].to_csv(newname + '.csv', encoding='utf-8')

if __name__ == '__main__':
    name = "balancedCombinedResult"
    newname = "balancedCombinedProcessed"
    cleanUpText(name,newname)
