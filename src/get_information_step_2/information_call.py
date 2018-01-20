
import urllib.request
import time
import csv
import os
import json
import urllib
import pickle
import pymongo
from pymongo import MongoClient
import sys

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
    time.sleep(.25)
    url = "http://api.walmartlabs.com/v1/items?ids={0}&apiKey={1}&format=json".format(items, apiKey)
    data = json.load(urllib.request.urlopen(url))
    return data

def pickle_json_data_dict(pickle_name, data):
    filename = pickle_name + ".pickle"
    print("...saving pickle")
    tmp = open(filename,'wb')
    pickle.dump(data,tmp)
    print("pickle saved")


def run_all(filepath, collocetion, start, end, segment_num=20):
    id_list = open_csv_with_ids(filepath)
    print ("id_list imported successfully.")
    unique_id_list= remove_duplicates(id_list)
    print ("unique list created successfully.")
    segmented_ids = segment_and_concat_id_list_for_api(unique_id_list, segment_num)
    print ("segmented list created successfully, {} total segments".format(len(segmented_ids)))
    short_list = segmented_ids[start:end]
    for number, segment in enumerate(short_list):
        try:
            data_dict = load_json_from_api(segment, SECRET_KEY)
        except:
            ###Error handles HTML timeout from walmarts gateat error 504
            time.sleep(2)
            print('Error Handled')
            try:
                data_dict = load_json_from_api(segment, SECRET_KEY)
            except:
                print ('400 error')
                continue

        for item in data_dict['items']:
            collection.replace_one(
                {'itemId': item['itemId']},
                item, upsert=True)
        print ('segment complete {}/{}, there are {} items in the collection.'
                .format(number + start, len(segmented_ids), collection.count()))


if __name__ == '__main__':
    client = MongoClient()
    db = client.capstone_project
    collection = db.product_id_returns
    start = int(sys.argv[1])
    end = int(sys.argv[2])

    run_all('../../data/all_ids', collection, start, end)

    # results = collection.find({'$or': [{'availableOnline' : False}, {'availableOnline' : True}]},
    #                                 {'itemId': 1,
    #                                  'shortDescription' : 1,
    #                                  'imageEntities.thumbnailImage' : 1,
    #                                  'categoryPath' : 1,
    #                                  'name' : 1})
    # for result in results:
    #     pprint.pprint(result)
    # print (results.count())
