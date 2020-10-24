import preprocessor as p
from ekphrasis.classes.segmenter import Segmenter
from nltk.tokenize import TweetTokenizer
import nltk
import re
from nltk.corpus import stopwords
import pandas as pd
import csv


nltk.download
nltk.download('wordnet')
nltk.download('stopwords')


def readfile(filepath):
    return pd.read_csv(filepath, encoding='utf-8', memory_map=True)


def splitUpTweets(data, corpus):
    a = []
    if (data != a):
        listToStr1 = ' '.join([str(elem.lower()) for elem in data])
        return corpus.segment(listToStr1)


def remove_punctuation(words):
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words


# df = data drame
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    df_pd = readfile(
        "crowdflower-brands-and-product-emotions/data/judge_1377884607_tweet_product_company.csv")

    # remove nulls and reset index
    df_pd = df_pd.dropna(
        subset=['tweet_text', 'is_there_an_emotion_directed_at_a_brand_or_product'])
    df_pd.reset_index(drop=True, inplace=True)

    # extract and decompound hashtag
    seg_tw = Segmenter(corpus="twitter")
    df_pd['hashtag'] = df_pd['tweet_text'].apply(
        lambda x: re.findall(r"#(\w+)", x))
    df_pd['seghash'] = df_pd['hashtag'].apply(
        lambda x: splitUpTweets(x, seg_tw))

    # TODO: extract emojis and smileys

    # remove url, hashtags, mentions, RT and FV words, emojis, smileys
    for v, i in enumerate(df_pd['tweet_text']):
        df_pd.loc[v, "text"] = p.clean(i)
        df_pd.loc[v, "text"] = p.clean(i.partition("RT")[0])
        # df_pd.loc[v, "text"] = p.clean(i.partition("FV")[0])

    # remove numbers
    data = df_pd['text'].astype(str).str.replace('\d+', '')
    # data = re.sub('[^0-9a-zA-Z]+', ' ', (str)df_pd['text'])

    # lowercase
    lower_case = data.str.lower()

    # TODO: extract ?? and !! (can have more than 2)

    # lemamatization and tokenization
    lematizer = nltk.stem.WordNetLemmatizer()
    tokenizer = TweetTokenizer()

    words = lower_case.apply(
        lambda x: [(lematizer.lemmatize(w)) for w in tokenizer.tokenize(x)])

    # remove punctation
    words = words.apply(remove_punctuation)

    # remove stop words
    stop_words = set(stopwords.words('english'))
    no_stop_words = words.apply(
        lambda x: [item for item in x if item not in stop_words])

    # display
    pd.set_option('display.max_columns', None)
    # print(df_pd.head(50))
    #print("Red sus")
data.to_csv('results.csv', encoding='utf-8', header=False)
