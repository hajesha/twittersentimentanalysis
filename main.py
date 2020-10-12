import nltk
from nltk.corpus import stopwords
import pandas as pd

def readfile(filepath):
    return pd.read_csv(filepath)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    df_pd = readfile("crowdflower-brands-and-product-emotions/data/judge_1377884607_tweet_product_company.csv")
    print(df_pd.head(5))