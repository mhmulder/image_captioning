import numpy as np
import pandas as pd
from pymongo import MongoClient
import re
from string import punctuation
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
import os
from collections import Counter

'''
This script does string processing and gets the text ready for analysis.
'''


def count_dict(list_of_strings):
    """
    Creates a counter dictionary, that counts each word in the corpus of
    descriptions.

    paramters:
    --------------------------
    list_of_strings (list) -> list of descriptions

    returns:
    --------------------------
    my_counter (dict) -> A dictionary with words and their corresponding counts
    """
    PUNCT_DICT = {ord(punc): None for punc in punctuation}
    string = " ".join(list_of_strings)
    string = string.translate(PUNCT_DICT).lower()
    tokens = string.strip(" ").split(" ")

    my_counter = Counter(tokens)

    return my_counter


def process_string(string, count_dict):
    """
    Cleaning function, that strips punctuation, makes everything lowercase,
    adds a '<end>' and '<start>' token, removes tokens that start with numbers,
    removes stopwords, removes words less than 3 characters, removes words that
    appear fewer than 5 times.

    paramters:
    --------------------------
    string (str) -> A string to process
    count_dict (dict) -> A dictionary with words and their corresponding counts

    returns:
    --------------------------
    (str) -> A cleaned and processed string
    """
    PUNCT_DICT = {ord(punc): None for punc in punctuation}
    string = string.translate(PUNCT_DICT).lower()
    tokens = string.strip(" ").split(" ")

    clean_words = ['<start>']
    for word in tokens:
        if len(word) < 3:
            continue
        # regex remove words starting with numbers
        matches = re.match('\d', word)
        if matches:
            continue
        if count_dict[word] < 5:
            continue
        clean_words.append(word)
    clean_words.append('<end>')

    # can add additional stop words here
    stopwords = set(list(ENGLISH_STOP_WORDS) + ['deep', 'high', 'wide'])

    return ' '.join(word for word in clean_words if word not in stopwords)


def run_query(query, collection):
    """
    Queries a Mongo collection for something and returns the results.

    paramters:
    --------------------------
    query (str) -> A query to run through MongoDB
    collection (MongoDB collection) -> A collection containing the
        information for all product ID's

    returns:
    --------------------------
    (MongoDB cursor) -> A cursor containing the results of the query
    """
    rgx = re.compile(query, re.IGNORECASE)
    results = collection.find({'$and': [{'categoryPath': rgx},
                              {'imageEntities.thumbnailImage':
                              {'$exists': True}}]})
    return results


def df_cleaning_func(df, count_dict):
    """
    Really simple df cleaning function, removes brand names from descriptions,
    and then runs the process string function.

    paramters:
    --------------------------
    df (pandas DataFrame) -> A dataframe with strings to process
    count_dict (dict) -> A dictionary with words and their corresponding counts

    returns:
    --------------------------
    df (pandas DataFrame) -> A dataframe with processed strings
    """
    # currently short may add_more
    df.shortDescription = df['shortDescription'].str.strip('&lt;p&gt;').head()

    # remove brand from name
    name_striped = [name.replace(brand, '') for name, brand in zip(
                    df['name'].astype('str'), df['brandName'].astype('str'))]

    # function to clean strings
    clean_striped = [process_string(desc, count_dict) for desc in name_striped]

    # add column back to df
    df['cleanDescription'] = clean_striped

    return df


def caption_mapping(filename, images_dir, item_desc_dict):
    """
    This function creates a csv with image names and their descriptions.
    This document essentially links everything together.

    paramters:
    --------------------------
    filename (str) -> The name or path of the resulting csv
    images_dir (list) -> A list of all image names
    item_desc_dict(dict) => A dictionary of descriptions and the image names

    returns:
    --------------------------
    None, a csv is created in the directory associated with the filename
    """
    with open(filename, 'w') as w:
        for pic_name in images_dir:
            itemId = pic_name.split("_")[0]
            try:
                desc = item_desc_dict[int(itemId)]
            except:
                continue
            w.write(pic_name + "," + desc + "\n")
            w.flush()

if __name__ == '__main__':
    # load mongo client in the cloud
    mongo_password = os.environ['capstone_mongo']
    client = MongoClient(mongo_password)
    db = client.capstone_project
    collection = db.product_id_returns
    print('Mongol loaded!')

    # query mongo for all furniture items and return all items with images
    results = run_query('HOME/Furniture.*', collection)
    print('{} results found'.format(results.count()))

    # convert data into a pandas dataframe
    df = pd.DataFrame(list(results))
    print('DataFrame created!')
    list_of_strings = list(df['name'].values)

    my_counter = count_dict(list_of_strings)

    # clean df
    cleaned_df = df_cleaning_func(df, my_counter)
    print('DataFrame cleaned!')
    # put descriptions in a dictionary
    item_desc_dict = pd.Series(cleaned_df['cleanDescription'].values,
                               index=cleaned_df['itemId']).to_dict()
    print('Dictionaries made!')
    # get all image names
    images_dir = os.listdir("../../data/pics/")
    print('{} pictures found!'.format(len(images_dir)))
    # create csv for image mapping later
    caption_mapping('../../data/caption_mapping.txt',
                    images_dir,
                    item_desc_dict)

    # load csv, unnesciasary to convert back but helpful to have the file
    image_description_pair = pd.read_csv('../../data/caption_mapping.txt',
                                         names=['image_name', 'description'])

    # create train/test splits
    id_pair_8k_train = image_description_pair.iloc[0:8000]
    id_pair_8k_test = image_description_pair.iloc[8000:10000]
    id_pair_full_train = image_description_pair.iloc[:250000]
    id_pair_full_test = image_description_pair.iloc[250000:]

    id_pair_8k_test.to_csv('../../data/image_description_pair_8k_test.csv')
    id_pair_8k_train.to_csv('../../data/image_description_pair_8k_train.csv')
    id_pair_full_train.to_csv('../../data/image_description_full_train.csv')
    id_pair_full_test.to_csv('../../data/image_description_full_test.csv')
    print('Finished')
