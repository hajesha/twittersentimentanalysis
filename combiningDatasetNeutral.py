import pandas as pd
from sklearn.utils import resample
import matplotlib.pyplot as plt

def readfile(filepath):
    return pd.read_csv(filepath, encoding='utf-8', memory_map=True)


def combineDataset(name):
    # first set
    df_pd = readfile(
        "dataset/judge_1377884607_tweet_product_company.csv")

    # second set
    filelist = ['dataset/train_text.txt', 'dataset/train_labels.txt']
    newpanda1 = pd.read_csv('dataset/train_text.txt',
                            delimiter='\n', names=['text'])
    newpanda2 = pd.read_csv('dataset/train_labels.txt',
                            delimiter='\n', names=['emotion'])

    newpanda3 = pd.read_csv('dataset/test_text.txt',
                            delimiter='\n', names=['text'])
    newpanda4 = pd.read_csv('dataset/test_labels.txt',
                            delimiter='\n', names=['emotion'])

    newpanda5 = pd.read_csv('dataset/val_text.txt',
                            delimiter='\n', names=['text'])
    newpanda6 = pd.read_csv('dataset/val_labels.txt',
                            delimiter='\n', names=['emotion'])
    newpanda = pd.concat(
        [
            newpanda1.reset_index(drop=True),
            newpanda2.reset_index(drop=True),
        ],
        axis=1,
        ignore_index=True,
    )
    newpanda = newpanda.append(pd.concat(
        [
            newpanda3.reset_index(drop=True),
            newpanda4.reset_index(drop=True),
        ],
        axis=1,
        ignore_index=True,
    ))

    newpanda = newpanda.append(pd.concat(
        [
            newpanda5.reset_index(drop=True),
            newpanda6.reset_index(drop=True),
        ],
        axis=1,
        ignore_index=True,
    ))

    newpanda.columns = ['text', 'emotion']

    # final output
    output_df = pd.DataFrame()

    # remove neutral values
    df_pd = df_pd[df_pd.is_there_an_emotion_directed_at_a_brand_or_product != "I can't tell"]
    # df_pd = df_pd[df_pd.is_there_an_emotion_directed_at_a_brand_or_product != "No emotion toward brand or product"]
    # newpanda = newpanda[newpanda.emotion != '1']

    # parse labels
    newpanda['emotion'] = newpanda['emotion'].replace({1: 0, 2.0: 1, 0.0: -1})
    df_pd['emotion'] = df_pd['is_there_an_emotion_directed_at_a_brand_or_product'].replace({"Positive emotion": 1, "Negative emotion": -1, "No emotion toward brand or product": 0})


    # combine all of the datasets together
    output_df = output_df.append(newpanda)
    output_df = output_df.append(df_pd[["tweet_text", "emotion"]])
    output_df = output_df.drop(columns="tweet_text")
    # drop any null rows
    output_df = output_df.dropna(subset=['text'])
    output_df.reset_index(drop=True, inplace=True)

    # save to a csv
    output_df[['text','emotion']].to_csv(name + '.csv', encoding='utf-8')


def balanceDataset(name):
    unbalancedSet = readfile(name + '.csv')
    df_majority = unbalancedSet[unbalancedSet.emotion == 1]
    df_minority = unbalancedSet[unbalancedSet.emotion == -1]
    df_minority2 = unbalancedSet[unbalancedSet.emotion == 0]
    majority_equal = resample(df_majority,
                                     replace=False,     # sample with replacement
                                     n_samples=len(df_minority))

    df_minority2 = resample(df_minority2,
                              replace=False,  # sample with replacement
                              n_samples=len(df_minority))
    equaldataframe = pd.concat([majority_equal, df_minority, df_minority2])
    equaldataframe[['text','emotion']].to_csv(name + '.csv', encoding='utf-8')


def randomDownsize(name, number):
    dataframe = readfile(name + '.csv')
    downsizeddataframe = resample(dataframe, n_samples=number)
    downsizeddataframe.to_csv(name + '.csv', encoding='utf-8')


def plot(name):
    dataframe = readfile(name + '.csv')
    plt.pie(dataframe.emotion.value_counts(), labels=["positive", "negative", "neutral"], autopct='%1.1f%%')
    print(dataframe.emotion.value_counts())
    plt.show()


if __name__ == '__main__':
    name = "balancedDataNeutral"
    combineDataset(name)
    balanceDataset(name)
    # randomDownsize(name, 200)
    # plot(name)


