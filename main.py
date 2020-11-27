import pandas as pd
from nltk.corpus import stopwords
import re
import nltk
from nltk.tokenize import TweetTokenizer
from ekphrasis.classes.segmenter import Segmenter
import preprocessor as p
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download
nltk.download('wordnet')
nltk.download('stopwords')


def readfile(filepath):
    return pd.read_csv(filepath, encoding='utf-8', memory_map=True)


def splitUpTweets(data, corpus):
    a = []
    if (data != a):
        listToStr1 = ' '.join([str(elem.lower()) for elem in data])
        return corpus.segment(listToStr1).split()


def remove_links(words):
    new_words = []
    for word in words:
        new_word = re.sub(' ', '-----', word)
        new_words.append(new_word)
    return new_words


def remove_punctuation(words):
    new_words = []
    for word in words:
        new_word = re.sub('[^a-z\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words


def get_words(message): return [i for item in message for i in item.split()]


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    df_pd = readfile(
        "crowdflower-brands-and-product-emotions/data/judge_1377884607_tweet_product_company.csv")
    #
    # df_pd2 = pd.read_csv(
    #     "dataset/processedNegative.csv", index_col=None)
    #
    # df_pd3 = pd.read_csv(
    #     "dataset/processedPositive.csv",  index_col=None)
    stop_words = set(stopwords.words('english'))
    filelist = ['dataset/train_text.txt', 'dataset/train_labels.txt']
    newpanda1 = pd.read_csv('dataset/train_text.txt' , delimiter='\n', names=['original_text'])
    newpanda3 = pd.read_csv('dataset/train_labels.txt' , delimiter='\n', names=['emotion'])

    newpanda = pd.concat(
        [
            newpanda1.reset_index(drop=True),
            newpanda3.reset_index(drop=True),
        ],
        axis=1,
        ignore_index=True,

    )

    newpanda.columns = ['original_text', 'emotion']
    #
    # df_pd2 = df_pd2.T
    # df_pd3 = df_pd3.T
    # df_pd2 = df_pd2.set_index()
    # df_pd3 = df_pd3.set_index()
    # # remove nulls and reset index
    # # df_pd = df_pd.dropna(
    # #     subset=['tweet_text', 'is_there_an_emotion_directed_at_a_brand_or_product'])
    #
    # # df_pd2 = df_pd2.dropna()
    # # df_pd3 = df_pd3.dropna()
    #
    # df_pd2[1] = "-1"
    # df_pd3[1] = "1"
    #
    # df_pd2.columns = ['original_text', 'emotion']
    # df_pd3.columns = ['original_text', 'emotion']
    # # df_pd2.reset_index(drop=True, inplace=True)
    # df_pd3.reset_index(drop=True, inplace=True)

    df_pd = df_pd[df_pd.is_there_an_emotion_directed_at_a_brand_or_product != "I can't tell"]
    df_pd = df_pd[df_pd.is_there_an_emotion_directed_at_a_brand_or_product != "No emotion toward brand or product"]
    # df_pd.reset_index(drop=True, inplace=True)

    newpanda = newpanda[newpanda.emotion != '1']
    newpanda['emotion'] = newpanda['emotion'].replace(
        {2: 1, 0: -1})
    output_df = pd.DataFrame()
    output_df['original_text'] = df_pd['tweet_text']

    output_df['emotion'] = df_pd['is_there_an_emotion_directed_at_a_brand_or_product'].replace(
        {"Positive emotion": 1, "Negative emotion": -1})

    output_df = output_df.append(newpanda)



    output_df = output_df.dropna(subset=['original_text'])
    output_df.reset_index(drop=True, inplace=True)

    # extract and decompound hashtag
    seg_tw = Segmenter(corpus="twitter")
    output_df['hashtags'] = output_df['original_text'].apply(
        lambda x: re.findall(r"#(\w+)", x))

    # output_df['emotion'] = df_pd['is_there_an_emotion_directed_at_a_brand_or_product'].replace(
    #     {"Positive emotion": 1, "Negative emotion": -1, "No emotion toward brand or product": 0})




    seghash = output_df['hashtags'].apply(lambda x: splitUpTweets(x, seg_tw))

    for i in range(len(seghash)):
        if seghash[i] is not None:
            seghash[i] = list(
                filter(lambda a: ((a not in stop_words) & (a != "_")), seghash[i]))

    output_df['hashtags'] = seghash

    # extract emojis and smileys
    output_df['emojis'] = output_df['original_text'].apply(
        lambda x: re.findall(r"([:;=](?:')?(?:-)?[)DPO])", x))

    output_df['exaggerate_punctuation'] = output_df['original_text'].apply(
        lambda x: re.search("[?!.]{2,}", x) is not None)

    # remove url, hashtags, mentions, RT and FV words, emojis, smileys
    for v, i in enumerate(output_df['original_text']):
        output_df.loc[v, "text"] = p.clean(i)
        output_df.loc[v, "text"] = p.clean(i.partition("RT")[0])
        # df_pd.loc[v, "text"] = p.clean(i.partition("FV")[0])

    # lowercase
    lower_case = output_df['original_text'].str.lower()

    # lemamatization and tokenization
    lematizer = nltk.stem.WordNetLemmatizer()
    tokenizer = TweetTokenizer()

    words = lower_case.apply(
        lambda x: [(lematizer.lemmatize(w)) for w in tokenizer.tokenize(x)])

    # pos = df_pd['tweet_text'].apply(get_pos)
    # pos = lower_case.apply(get_pos)
    # remove punctation
    # df_pd['pos_tag'] = df_pd['tweet_text'].apply(
    # lambda x: [i for item in x for i in item.split()])

    # s = df_pd['text']
    # for v, i in enumerate(df_pd['pos_tag']):
    # df_pd.loc[v, "pos_tag"] = get_words(i)

    # print(df_pd['pos_tag'])
    pos = words.apply(
        lambda x: [i for item in x for i in item.split()])

    output_df['pos_tag'] = pos

    for value in output_df['pos_tag'].index:
        output_df['pos_tag'].loc[value] = nltk.pos_tag(
            output_df['pos_tag'].loc[value])

    words = words.apply(remove_punctuation)
    words = words.apply(remove_links)

    # remove stop words
    no_stop_words = words.apply(
        lambda x: [item for item in x if item not in stop_words])

    # print(output_df['pos_tag'])
    # display
    pd.set_option('display.max_columns', None)

    output_df['text'] = list(no_stop_words)
    # print(df_pd.head(50))
    # output the files
    output_df.to_csv('results.csv', encoding='utf-8')
