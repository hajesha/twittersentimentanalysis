Please use Python 3.8

To download the data set 
- Dataset 1: 
    - download dataset from this link https://data.world/crowdflower/brands-and-product-emotions
    - extract file "judge_1377884607_tweet_product_company.csv" from data file 
    - place into dataset file under root 
    
- Dataset 2: 
    - download the following files from this github and place it under the dataset folder as well
        - test_labels.txt
        - test_text.txt
        - train_labels.txt
        - train_text.txt
        - val_labels.txt
        - val_text.txt
        
- The file structure should look like this 
    -> finalproject
        -> dataset 
            -> judge_1377884607_tweet_product_company.csv
            -> test_labels.txt
            -> test_text.txt
            -> train_labels.txt
            -> train_text.txt
            -> val_labels.txt
            -> val_text.txt
  
  
- To Combine the datasets
    - make sure that all datasets are properly downloaded
    - run combineDataset.py  
    
- To Pre-Process text
    - pip install tweet-preprocessor
    - run preprocessing.py