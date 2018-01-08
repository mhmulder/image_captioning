import requests
from bs4 import BeautifulSoup
from urllib.request import urlopen
import numpy as np
import pandas as pd
import re
import time
import csv
import os
import json
import urllib
import pickle
import pymongo
from pymongo import MongoClient
import pprint
from bson import json_util
import json
import pickle
import pprint
from collections import defaultdict
import sys
import glob
import alert_program


def run_query(collection):
    rgx = re.compile('HOME/Furniture/Kitchen & Dining Furniture.*', re.IGNORECASE)
    new_results = collection.find({'$and': [{'categoryPath': rgx}, {'imageEntities.thumbnailImage' : {'$exists': True}}]},
                                    {'itemId': 1,
                                     'categoryPath' : 1,
                                     'imageEntities.thumbnailImage' : 1
                                    })
    return new_results

def append_to_csv(filepath, old_list, new_list):
    new_img_list = list(set(old_list + new_list))

    with open(filepath, 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(new_img_list)
    print("File saved as {}".format(filepath))

def read_csv(filepath):
    with open (filepath, 'r') as mycsv:
        images = csv.reader(mycsv, delimiter = ',')
        img_list = [img_num for img_num in images][0]
    return img_list

def create_img_dict(filepath, verbose=0):
    img_list = read_csv(filepath)
    # print(img_list)
    img_dict = defaultdict(list)
    high_num = 0
    for elem in img_list:
        # test = elem.strip('.jpg').split('_')
        img_num, img_product = elem.strip('.jpg').split('_')
        img_num = int(img_num)
        img_dict[img_product].append(img_num)
        if img_num > high_num:
            high_num = img_num
    if verbose != 0:
        print ('Image_dict_created')
    return img_dict, high_num, img_list


def get_photos(results, filepath, start, end):
    img_dict, high_num, old_list = create_img_dict(filepath)
    img_num = high_num
    img_list = []
    for i in range(start, end):
        if i % 10 == 0:
            print ('Now downloading pictures from product number {}'.format(i))
            append_to_csv(filepath, old_list, img_list)
        item_id = results[i]['itemId']
        if str(item_id) in img_dict:
            continue
        for item in results[i]['imageEntities']:
            time.sleep(0.2)
            img_num += 1
            url = item['thumbnailImage']
            filename = '{:06}_{}.jpg'.format(img_num,item_id)
            file_path = 'pics/{}'.format(filename)
            urllib.request.urlretrieve(url, file_path)
            img_list.append(filename)
    print ('{} images downloaded'.format(img_num-high_num))

def get_ids_from_directory():
    print (os.getcwd())
    img_list = []
    files = glob.glob(os.getcwd() + "/pics/"+"*.jpg")
    for fle in files:
        clean = fle.split('/')[-1]
        img_list.append(clean)
    append_to_csv('pics/img_list.csv', img_list, img_list)
    return (files)


if __name__ == '__main__':
    start = int(sys.argv[1])
    end = int(sys.argv[2])
    client = MongoClient()
    db = client.capstone_project
    collection = db.product_id_returns
    results = run_query(collection)
    try:
        get_photos(results, 'pics/img_list.csv', start, end)
    except:
        alert_program.send_end_alert('capstone', 'You have an error!')
    # files = get_ids_from_directory()
    # print (len(files))
