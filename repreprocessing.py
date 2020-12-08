import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from ekphrasis.classes.segmenter import Segmenter
import preprocessor as p
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('wordnet')

def readfile(filepath):
    return pd.read_csv(filepath, encoding='utf-8', memory_map=True)

def remove_links(words):
    new_words = []
    for word in words:
        if word not in stop_words:
            withoutlink = re.sub(' ', '-----', word)
            cleanWord = p.clean(withoutlink)
            new_word = re.sub('[^a-z\s]', '', cleanWord)
            if new_word != '' and new_word != "-----" and new_word != "rt":
                new_words.append(new_word)
    return new_words


def splitUpTweets(data, corpus):
    a = []
    if (data != a):
        listToStr1 = ' '.join([str(elem.lower()) for elem in data])
        return corpus.segment(listToStr1).split()


if __name__ == '__main__':

    # Initialize all the dataframes
    df_pd = readfile(
        "crowdflower-brands-and-product-emotions/data/judge_1377884607_tweet_product_company.csv")


    output_df = pd.DataFrame()
    stop_words = set(stopwords.words('english'))

    # Remove values where there is discrepency or neutral emotions
    df_pd = df_pd[df_pd.is_there_an_emotion_directed_at_a_brand_or_product != "I can't tell"]
    df_pd = df_pd[df_pd.is_there_an_emotion_directed_at_a_brand_or_product != "No emotion toward brand or product"]

    # Convert to ints
    df_pd['emotion'] = df_pd['is_there_an_emotion_directed_at_a_brand_or_product'].replace({"Positive emotion": 1, "Negative emotion": -1})
    # df_pd['emotion'] = df_pd['is_there_an_emotion_directed_at_a_brand_or_product'].replace({"Positive emotion": 1,"No emotion toward brand or product" : 0 ,"Negative emotion": -1})
    df_pd = df_pd.dropna(subset=['tweet_text'])
    # Extracthashtags
    seg_tw = Segmenter(corpus="twitter")
    df_pd['hashtags'] = df_pd['tweet_text'].apply(lambda x: re.findall(r"#(\w+)", x))

    # # decompound hashtags
    # seghash = df_pd['hashtags'].apply(lambda x: splitUpTweets(x, seg_tw))
    # seghash.reset_index(drop=True, inplace=True)
    # # Remove stop words in segmented tweet
    # for i in range(len(seghash)):
    #     if seghash[i] is not None:
    #         seghash[i] = list(
    #             filter(lambda a: ((a not in stop_words) & (a != "_")), seghash[i]))
    # output_df['hashtags'] = seghash

    # lemamatization and tokenization
    lematizer = nltk.stem.WordNetLemmatizer()
    tokenizer = TweetTokenizer()
    output_df['text'] = df_pd['tweet_text'].apply(
        lambda x: [(lematizer.lemmatize(w)) for w in tokenizer.tokenize(x.lower())])

    output_df['emotion'] = df_pd['emotion']
    output_df = output_df.dropna(subset=['text'])
    # CLean up the text
    output_df['text'] = output_df['text'].apply(remove_links)

    output_df = output_df.dropna(subset=['text'])
    output_df.reset_index(drop=True, inplace=True)

    # second set
    filelist = ['dataset/train_text.txt', 'dataset/train_labels.txt']
    newpanda1 = pd.read_csv('dataset/train_text.txt',
                            delimiter='\n', names=['original_text'])
    newpanda3 = pd.read_csv('dataset/train_labels.txt',
                            delimiter='\n', names=['emotion'])

    newpanda = pd.concat(
        [
            newpanda1.reset_index(drop=True),
            newpanda3.reset_index(drop=True),
        ],
        axis=1,
        ignore_index=True,

    )

    newpanda.columns = ['text', 'emotion']
    newpanda = newpanda[newpanda.emotion != '1']
    newpanda['emotion'] = newpanda['emotion'].replace({2.0: 1, 0.0: -1})

    # newpanda['hashtags'] = newpanda['text'].apply(lambda x: re.findall(r"#(\w+)", x))
    # seghash = newpanda['hashtags'].apply(lambda x: splitUpTweets(x, seg_tw))
    # seghash.reset_index(drop=True, inplace=True)
    # # Remove stop words in segmented tweet
    # for i in range(len(seghash)):
    #     if seghash[i] is not None:
    #         seghash[i] = list(
    #             filter(lambda a: ((a not in stop_words) & (a != "_")), seghash[i]))

    # lemamatization and tokenization
    newpanda['text'] = newpanda['text'].apply(
        lambda x: [(lematizer.lemmatize(w)) for w in tokenizer.tokenize(x.lower())])

    newpanda = newpanda.dropna(subset=['text'])
    # CLean up the text
    newpanda['text'] = newpanda['text'].apply(remove_links)

    newpanda = newpanda.dropna(subset=['text'])
    newpanda.reset_index(drop=True, inplace=True)

    output_df = output_df.append(newpanda)
    output_df = output_df.dropna(subset=['text'])
    output_df.reset_index(drop=True, inplace=True)
    output_df.to_csv('resultsHK.csv', encoding='utf-8')