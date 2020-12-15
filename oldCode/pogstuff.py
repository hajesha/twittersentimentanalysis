import nltk as nltk
import pandas as pd
from ast import literal_eval


def readfile(filepath):
    return pd.read_csv(filepath, encoding='utf-8', memory_map=True)

if __name__ == '__main__':

    # Initialize all the dataframes
    df_pd = readfile("resultsupscaledHKSSSS.csv")

    df_pd['pos_tag'] =  df_pd.text.apply(literal_eval)

    # df_pd['pos_tag']= df_pd['pos_tag'].apply(nltk.pos_tag( df_pd['pos_tag']))
    df_pd['pos_tag']= df_pd['pos_tag'].apply(lambda x: [nltk.pos_tag(x)])
    df_pd['pos_tag']= df_pd['pos_tag'].apply(lambda z: [x[0] for x in z[0] if x[1] in ("VB", "VBD", "VBP", "VBZ", "JJ", "JJR", "JJS", "NN", "NNS", "NNPS", "RB", "RBR", "RBS")])

    # df_pd['pos_tag'] = list([x for (x,y) in df_pd['pos_tag'] if y not in ("VB", "VBD", "VBP", "VBZ", "JJ", "JJR", "JJS", "NN", "NNS", "NNPS", "RB", "RBR", "RBS")])
    df_pd.to_csv('resultsPOG.csv', encoding='utf-8')