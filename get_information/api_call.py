import requests
from bs4 import BeautifulSoup
from urllib.request import urlopen as uReq
import numpy as np
import pandas as pd
import re
import time
import csv
import os
import json
import urllib
import pickle

SECRET_KEY = os.environ["WM_SECRET_ACCESS_KEY"]

def open_csv_with_ids (filepath):
    with open (filepath, 'r') as mycsv:
        ids = csv.reader(mycsv, delimiter = ',')
        id_list = [id_num for id_num in ids][0]
    return id_list

def remove_duplicates (id_list):
    no_dupes_id_list = list(set(id_list))
    return no_dupes_id_list

def segment_and_concat_id_list_for_api(id_list, segment_num):
    concat_ids_for_api = []
    if len(id_list) % segment_num == 0:
        end = int(len(id_list)/segment_num)
    else:
        end = int(len(id_list)/segment_num + 1)
    segmented_ids = []
    for i in range (0, end):
        if i == end -1:
            segmented_ids.append(id_list[i*segment_num:])
        else:
            segmented_ids.append(id_list[i*segment_num:i*segment_num+segment_num])
    for segment in segmented_ids:
        concat_ids_for_api.append(",".join(segment))
    return concat_ids_for_api

def load_json_from_api(items, apiKey):
    time.sleep(2)
    url = "http://api.walmartlabs.com/v1/items?ids={0}&apiKey={1}&format=json".format(items, apiKey)
    data = json.load(urllib.request.urlopen(url))
    return data

def pickle_json_data_dict(pickle_name, data):
    filename = pickle_name + ".pickle"
    print("...saving pickle")
    tmp = open(filename,'wb')
    pickle.dump(data,tmp)
    print("pickle saved")

def run_all(filepath, segment_num):
    id_list = open_csv_with_ids(filepath)
    print ("id_list imported successfully")
    unique_id_list= remove_duplicates(id_list)
    print ("unique list created successfully")
    segmented_ids = segment_and_concat_id_list_for_api(unique_id_list, segment_num)
    print ("segmented list created successfully, {} total segments".format(len(segmented_ids)))

    for number, segment in enumerate(segmented_ids):
        if number == 0:
            data_dict = load_json_from_api(segment, SECRET_KEY)
            print ('segment complete {}/{}'.format(number + 1, len(segmented_ids)))
        else:
            temp_dict = load_json_from_api(segment, SECRET_KEY)
            for item in temp_dict['items']:
                data_dict['items'].append(item)
            print ('segment complete {}/{}'.format(number + 1, len(segmented_ids)))
    pickle_json_data_dict('200_lines', data_dict)

if __name__ == '__main__':
    # id_list = open_csv_with_ids('../UPC_scraping/5_pages_product_ids')
    # unique_id_list= remove_duplicates(id_list)
    # segmented_ids = segment_and_concat_id_list_for_api(unique_id_list,20)
    # print (segmented_ids)
    run_all('../UPC_scraping/5_pages_product_ids',20)
