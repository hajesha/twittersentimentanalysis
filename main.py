import nltk
import re
from nltk.corpus import stopwords
import pandas as pd

nltk.download
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.tokenize import TweetTokenizer
import preprocessor as p


def readfile(filepath):
    return pd.read_csv(filepath)


def remove_punctuation(words):
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    df_pd = readfile("crowdflower-brands-and-product-emotions/data/judge_1377884607_tweet_product_company.csv")

    # remove nulls
    df_pd = df_pd.dropna(subset=['tweet_text'])
    df_pd = df_pd.dropna(subset=['is_there_an_emotion_directed_at_a_brand_or_product'])

    # extract hashtag
    df_pd['hashtag'] = df_pd['tweet_text'].apply(lambda x: re.findall(r"#(\w+)", x))

    # TODO: extract emojis and smileys

    # remove url, hashtags, mentions, RT and FV words, emojis, smileys
    for v, i in enumerate(df_pd['tweet_text']):
        df_pd.loc[v, "text"] = p.clean(i)

    # remove numbers
    data = df_pd['text'].astype(str).str.replace('\d+', '')

    # lowercase
    lower_case = data.str.lower()

    # TODO: extract ?? and !! (can have more than 2)

    # lemamatization and tokenization
    words = lower_case.apply(
        lambda x: [(nltk.stem.WordNetLemmatizer().lemmatize(w)) for w in TweetTokenizer().tokenize(x)])

    # remove punctation
    words = words.apply(remove_punctuation)

    # remove stop words
    stop_words = set(stopwords.words('english'))
    no_stop_words = words.apply(lambda x: [item for item in x if item not in stop_words])

    df_pd['text'] = no_stop_words

    print(df_pd.head(20))
