import nltk
import re
from nltk.corpus import stopwords
import pandas as pd

def readfile(filepath):
    return pd.read_csv(filepath)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    df_pd = readfile("crowdflower-brands-and-product-emotions/data/judge_1377884607_tweet_product_company.csv")
    df_pd = df_pd.dropna(subset=['tweet_text'])
    df_pd = df_pd.dropna(subset=['is_there_an_emotion_directed_at_a_brand_or_product'])
    df_pd['hashtag'] = df_pd['tweet_text'].apply(lambda x: re.findall(r"#(\w+)", x))
    print(df_pd.head(10))