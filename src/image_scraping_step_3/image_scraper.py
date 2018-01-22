import requests
from urllib.request import urlopen
import numpy as np
import pandas as pd
import re
import csv
import os
import urllib
import pymongo
from pymongo import MongoClient

from collections import defaultdict
import sys
import glob
import alert_program


def run_query(collection):
    """
    Searches a MongoDB collection for all items in Home/furniture

    paramters:
    --------------------------
    collection (MongoDB collection) -> A collection containing the
    information for all product ID's

    returns:
    --------------------------
    (MongoDB cursor) -> A cursor containing the results of the query
    """
    rgx = re.compile('HOME/Furniture.*', re.IGNORECASE)
    new_results = collection.find({'$and':
                                  [{'categoryPath': rgx},
                                   {'imageEntities.thumbnailImage':
                                   {'$exists': True}}]
                                   },
                                  {'itemId': 1,
                                   'categoryPath': 1,
                                   'imageEntities.thumbnailImage': 1
                                   })
    return new_results


def append_to_csv(filepath, old_list, new_list):
    """
    Updates an image list and appends the name to a csv

    paramters:
    --------------------------
    filepath (str) -> A filepath for the csv
    old_list (list) -> An older list of image names
    new_list (list) -> A list of all new image names

    returns:
    --------------------------
    None, A csv is created according to the filepath
    """

    new_img_list = list(set(old_list + new_list))

    with open(filepath, 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(new_img_list)
    print("File saved as {}".format(filepath))


def read_csv(filepath):
    """
    Reads a csv in order to get image name

    paramters:
    --------------------------
    filepath (str) -> A filepath for the csv

    returns:
    --------------------------
    img_list (list) -> A list of image names
    """
    with open(filepath, 'r') as mycsv:
        images = csv.reader(mycsv, delimiter=',')
        img_list = [img_num for img_num in images][0]
    return img_list


def create_img_dict(filepath):
    """
    Creates an image dictionary where the product number is the key, and all
    associated images names are in a list as values.

    paramters:
    --------------------------
    filepath (str) -> A filepath for the csv to clean_words

    returns:
    --------------------------
    img_dict (dict) -> A dict containing product number(keys) and picture names
        (values)
    img_list (list) -> A list of all images contained in the dictionary
    """
    img_list = read_csv(filepath)
    # print(img_list)
    img_dict = defaultdict(list)
    for elem in img_list:
        # test = elem.strip('.jpg').split('_')
        img_product, img_num = elem.strip('.jpg').split('_')
        img_num = int(img_num)
        img_dict[img_product].append(img_num)
    return img_dict, img_list


def get_photos(results, filepath, start, end, csv_name):
    """
    Takes in results and an outgoing file path and scrapes the website
    for images.

    paramters:
    --------------------------
    results (MongoDB cursor) -> A cursor containing the results of a query
    filepath (str) -> The location of the csv name to append pictures to
    start (int) -> The number in the results list that the query should start
    end (int) -> The number in the results list that the query should end at
    csv_name (str) -> The name of the csv that contains the image and
        product id information


    returns:
    --------------------------
    None, Items get stored to a local folder called pics in this directory.
    """
    img_dict, old_list = create_img_dict(csv_name)
    img_num = 0
    img_list = []
    results = [result for result in results]
    for i in range(start, end):
        if i % 10 == 0:
            print('Now downloading pictures from product number {}/{}'.format(
                   i, len(results)))
            append_to_csv(filepath, old_list, img_list)
        item_id = results[i]['itemId']
        if str(item_id) in img_dict:
            continue
        for counter, item in enumerate(results[i]['imageEntities']):
            # time.sleep(0.2)
            img_num += 1
            url = item['thumbnailImage']
            filename = '{}_{:02}.jpg'.format(item_id, counter)
            file_path = 'pics/{}'.format(filename)
            try:
                urllib.request.urlretrieve(url, file_path)
            except:
                continue
            img_list.append(filename)
    print('{} total images in directory.'.format(img_num))
    append_to_csv(filepath, old_list, img_list)


def get_ids_from_directory():
    """
    Gets the picture ids from the pics folder in the working directory and
    appends them to a list and updates the csv. Only required when there is a
    mismatch in the image csv and image directory.

    paramters:
    --------------------------
    None, runs in working directory and gets all .jpgs


    returns:
    --------------------------
    files (list) -> A list of all .jpgs in pics folder of working directory.
    """
    print(os.getcwd())
    img_list = []
    files = glob.glob(os.getcwd() + "/pics/"+"*.jpg")
    for fle in files:
        clean = fle.split('/')[-1]
        img_list.append(clean)
    append_to_csv('pics/img_list.csv', img_list, img_list)
    return files


if __name__ == '__main__':

    # Pull in arguments from command line so multiple versions of this script
    # Can be run over different portion of the results query.
    start = int(sys.argv[1])
    end = int(sys.argv[2])

    # Instantiate mongo with secret password in cloud
    mongo_password = os.environ['capstone_mongo']
    client = MongoClient(mongo_password)
    db = client.capstone_project
    collection = db.product_id_returns
    results = run_query(collection)
    try:
        get_photos(results, 'pics/img_list.csv', start, end, 'img_list.csv')
    except:
        alert_program.send_end_alert('capstone', 'You have an error!')
