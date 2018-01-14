import numpy as np
import pandas as pd
from pymongo import MongoClient
import re
from string import punctuation
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
import os

def process_string(string):
    # Create stoplist
    STOPLIST = set(list(ENGLISH_STOP_WORDS) +['x'])

    # strip punctuation
    PUNCT_DICT = {ord(punc): None for punc in punctuation}
    string = string.translate(PUNCT_DICT)

    # tokenize
    tokens = string.strip(" ").split(" ")

    # remove stopwords
    no_stopwords_tokens = [token for token in tokens if token not in STOPLIST]

    # sort and join alphabetically (may be useful for NMT)
    return ' '.join(word for word in sorted(no_stopwords_tokens))


def run_query(query, collection):
    rgx = re.compile(query, re.IGNORECASE)
    results = collection.find({'$and': [{'categoryPath': rgx}, {'imageEntities.thumbnailImage' : {'$exists': True}}]})
    return results

def df_cleaning_func(df):
    # currently short may add_more
    df['shortDescription'] = df['shortDescription'].str.strip('&lt;p&gt;').head()

    # remove brand from name
    name_striped = [name.replace(brand, '') for name, brand in zip(df['name'].astype('str'), df['brandName'].astype('str'))]

    # function to remove stopwords, punctuation, and alphabitize words
    clean_striped = [process_string(desc) for desc in name_striped]

    df['cleanDescription'] = clean_striped

    return df


if __name__ == '__main__':
    ### to import pics
    ### $ sudo apt install awscli
    ### $ aws config
    ### test with $ aws s3 ls
    ### aws s3 sync s3://mm-capstone-images image_Scraping

    # load mongo client
    mongo_password = os.environ['capstone_mongo']
    client = MongoClient(mongo_password)
    db = client.capstone_project
    collection = db.product_id_returns


    # query mongo for all furniture items and return all items with images
    results = run_query('HOME/Furniture.*', collection)

    # convert data into a pandas dataframe
    df = pd.DataFrame(list(results))

    # clean df
    cleaned_df = df_cleaning_func(df)

    # copy the df
    df_to_Add_pics = df[['cleanDescription', 'itemId']].copy()
