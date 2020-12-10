import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import resample

def readfile(filepath):
    return pd.read_csv(filepath, encoding='utf-8', memory_map=True)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    ## Simple Plot
    # df_pd = readfile("resultsHK.csv").sample(frac=1,random_state=4)
    # plt.pie(df_pd.emotion.value_counts())
    # plt.show()

    df_pd = readfile("results.csv").sample(frac=1,random_state=4)
    plt.pie(df_pd.emotion.value_counts(),  labels=["positive","negative"], autopct='%1.1f%%')
    print(df_pd.emotion.value_counts())
    plt.show()
    df_majority = df_pd[df_pd.emotion == 1]
    df_minority = df_pd[df_pd.emotion == -1]
    # df_majority_resampled = resample(df_majority,
    #                                  replace=False,     # sample with replacement
    #                                  n_samples=8000,    # to match majority class
    #                                  random_state=123)
    # df_upsampled = pd.concat([df_majority_resampled, df_minority])
    df_upsampled = resample(df_pd, n_samples=10000, random_state=123)
    # df_upsampled.to_csv('resultsupscaled.csv', encoding='utf-8')
    df_upsampled.to_csv('resultsupscaledHKSSSS.csv', encoding='utf-8')
    plt.pie(df_upsampled.emotion.value_counts(),  labels=["positive","negative"], autopct='%1.1f%%')
    print(df_upsampled.emotion.value_counts())
    plt.show()

    #
    # df_majority = df_pd[df_pd.emotion == 1]
    # df_minority = df_pd[df_pd.emotion == -1]
    # print(df_pd.emotion.value_counts())
    # # df_minority_upsampled = resample(df_majority,
    # #                                  replace=False,     # sample with replacement
    # #                                  n_samples=7613,    # to match majority class
    # #                                  random_state=123)
    # #
    # # df_upsampled = pd.concat([df_minority_upsampled, df_minority])
    #
    # df_majority_sample = df_pd.loc[df_pd['emotion'] == 1].sample(n=7613, random_state=42)
    # df_sampled = pd.concat([df_majority_sample, df_minority])
    # print(df_sampled.emotion.value_counts())
    #
    # df_sampled.to_csv('resultsupscaled.csv', encoding='utf-8')