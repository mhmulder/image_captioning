import numpy as np
import pandas as pd
from pymongo import MongoClient
import re
from string import punctuation
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
import os
from PIL import Image
from PIL import ImageFile
from collections import Counter


def count_dict(list_of_strings):
    PUNCT_DICT = {ord(punc): None for punc in punctuation}
    string = " ".join(list_of_strings)
    string = string.translate(PUNCT_DICT).lower()
    tokens = string.strip(" ").split(" ")

    my_counter=Counter(tokens)

    return my_counter


def process_string(string, count_dict):
    PUNCT_DICT = {ord(punc): None for punc in punctuation}
    string = string.translate(PUNCT_DICT).lower()
    tokens = string.strip(" ").split(" ")

    clean_words = ['<start>']
    for word in tokens:
        if len(word) < 3:
             continue
        matches = re.match('\d', word)
        if matches:
            continue
        if count_dict[word] < 5:
            continue
        clean_words.append(word)
    clean_words.append('<end>')

    stopwords = set(list(ENGLISH_STOP_WORDS) + ['deep', 'high', 'wide'])

    return ' '.join(word for word in clean_words if word not in stopwords )


def run_query(query, collection):
    rgx = re.compile(query, re.IGNORECASE)
    results = collection.find({'$and': [{'categoryPath': rgx}, {'imageEntities.thumbnailImage' : {'$exists': True}}]})
    return results

def df_cleaning_func(df, count_dict):
    # currently short may add_more
    df['shortDescription'] = df['shortDescription'].str.strip('&lt;p&gt;').head()

    # remove brand from name
    name_striped = [name.replace(brand, '') for name, brand in zip(df['name'].astype('str'), df['brandName'].astype('str'))]

    # function to clean strings
    clean_striped = [process_string(desc, count_dict) for desc in name_striped]

    # add column back to df
    df['cleanDescription'] = clean_striped

    return df

def pic_2_df(pic_list, verbose = False):
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    new_df = pd.DataFrame(columns=['prod_id','pic_array'])
    i = 0
    for pic in pic_list:
        try:
            img = Image.open(pic)
            resize_img = img.resize([100,100])
        except:
            continue
        arr = np.array(resize_img)
        if arr.shape != (100,100,3):
            continue
        resize_img = img.resize([100,100])
        arr = np.array(resize_img)
        prod_id = re.search('(\d{5,12})', pic)[0]
        data = {'prod_id': prod_id, 'pic_array' : arr}
        new_df = new_df.append(data, ignore_index=True)
        i += 1
        if verbose and i % 10000 == 0:
            n = len(pic_list)
            print ('Finished {}/{} images.'.format(i,n))
    return new_df

def caption_mapping(filename, images_dir, item_desc_dict):
    with open(filename,'w') as w:
        for pic_name in images_dir:
            itemId = pic_name.split("_")[0]
#             print(itemId)
            try:
                desc = item_desc_dict[int(itemId)]
#                 print(desc)
            except:
                continue
            w.write(pic_name+","+desc+"\n")
            w.flush()

if __name__ == '__main__':
    ### to import pics
    ### $ sudo apt install awscli
    ### $ aws configure
    ### test with $ aws s3 ls
    ### aws s3 sync s3://mm-capstone-images image_Scraping

    # load mongo client
    mongo_password = os.environ['capstone_mongo']
    client = MongoClient(mongo_password)
    db = client.capstone_project
    collection = db.product_id_returns
    print ('Mongol loaded!')


    # query mongo for all furniture items and return all items with images
    results = run_query('HOME/Furniture.*', collection)
    print ('{} results found'.format(results.count()))
    # convert data into a pandas dataframe
    df = pd.DataFrame(list(results))
    print ('DataFrame created!')
    list_of_strings = list(df['name'].values)

    my_counter = count_dict(list_of_strings)

    # clean df
    cleaned_df = df_cleaning_func(df, my_counter)
    print ('DataFrame cleaned!')
    # put descriptions in a dictionary
    item_desc_dict = pd.Series(cleaned_df['cleanDescription'].values,index=cleaned_df['itemId']).to_dict()
    print ('Dictionaries made!')
    # get all image names
    images_dir = os.listdir("../../data/pics/")
    print ('{} pictures found!'.format(len(images_dir)))
    # create csv for image mapping later
    caption_mapping('../../data/caption_mapping.txt', images_dir, item_desc_dict)

    # load csv, unnesciasary to convert back but helpful to have the file
    image_description_pair = pd.read_csv('../../data/caption_mapping.txt', names = ['image_name', 'description'])


    # create train/test splits
    image_description_pair_8k_train = image_description_pair.iloc[0:8000]
    image_description_pair_8k_test = image_description_pair.iloc[8000:10000]
    image_description_pair_full_train = image_description_pair.iloc[:250000]
    image_description_pair_full_test = image_description_pair.iloc[250000:]

    image_description_pair_8k_test.to_csv('../../data/image_description_pair_8k_test.csv')
    image_description_pair_8k_train.to_csv('../../data/image_description_pair_8k_train.csv')
    image_description_pair_full_train.to_csv('../../data/image_description_pair_full_train.csv')
    image_description_pair_full_test.to_csv('../../data/image_description_pair_full_test.csv')
    print ('Finished')
