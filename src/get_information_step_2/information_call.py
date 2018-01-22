
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

"""
This script takes a list of product IDs,makes calls to an API and stores
the results in a MongoDB database. The database was originally stored locally
and then moved to the cloud when I was moving between computers. This script
can be run from the command line using start and end arguments. For example:

`python information_call.py 0 1000`

Segments the ID list and only scrape from the first 1000 elements. This allows
me to run multiple scraping scripts to cut down on time loss.

"""

SECRET_KEY = os.environ["WM_SECRET_ACCESS_KEY"]


def open_csv_with_ids(filepath):
    """
    Opens a csv and returns a list of the product ID's contained in that csv

    paramters:
    --------------------------
    filepath (str) -> the filepath of the folder

    returns:
    --------------------------
    id_list (list) -> A python list containing the product ID's for all
    elements.
    """
    with open(filepath, 'r') as mycsv:
        ids = csv.reader(mycsv, delimiter=',')
        id_list = [id_num for id_num in ids][0]
    return id_list


def remove_duplicates(id_list):
    """
    Takes the ID list removes the duplicates, and returns them in sorted order

    paramters:
    --------------------------
    id_list (list) -> A python list containing the product ID's for all
    elements.

    returns:
    --------------------------
    (list) -> A python list containing unique sorted product ID's
    """
    no_dupes_id_list = list(set(id_list))
    return sorted(no_dupes_id_list)


def segment_and_concat_id_list_for_api(id_list, segment_num=20):
    """
    Segments the ID list into the segment_num, which is defaulted to 20, per
    the limittions on the websites API.It then concatenates them into strings
    for use in the API

    paramters:
    --------------------------
    id_list (list) -> A python list containing the product ID's for all
    elements.

    segment_num (int) -> The number of unique ID's that can be passed per
    call to the API.

    returns:
    --------------------------
    concat_ids_for_api (list of strings) -> A list of concatenated ID's that
    can be passed to the API.
    """
    concat_ids_for_api = []
    if len(id_list) % segment_num == 0:
        end = int(len(id_list)/segment_num)
    else:
        end = int(len(id_list)/segment_num + 1)
    segmented_ids = []
    for i in range(0, end):
        if i == (end - 1):
            segmented_ids.append(id_list[i*segment_num:])
        else:
            segmented_ids.append(
                          id_list[i*segment_num:i*segment_num+segment_num])
    for segment in segmented_ids:
        concat_ids_for_api.append(",".join(segment))
    return concat_ids_for_api


def load_json_from_api(items, apiKey):
    """
    Calls the API with items and my secret key and returns a json dictionary
    with all elements of the items.

    paramters:
    --------------------------
    items (str) -> A concatenated segment of ID's that is n the proper form to
    be passed to the API.

    apiKey (str) -> A secret key that has to be used to access the API

    returns:
    --------------------------
    data(dict) -> A JSON dictionary of all elements, it is ready to be pushed
    into a Mongo Database
    """

    time.sleep(.25)
    url = ("http://api.walmartlabs.com/v1/items?ids={0}&apiKey={1}&format=json"
           .format(items, apiKey))
    data = json.load(urllib.request.urlopen(url))
    return data


def pickle_json_data_dict(pickle_name, data):
    """
    Dumps all of data into a pickle file, for compact and quick storage.

    paramters:
    --------------------------
    pickle_name (string) -> The name you'd like to name your pickle file

    data (dict) -> A JSON dictionary of product information

    returns:
    --------------------------
    None, outputs a pickle file to the directory specified in picklename
    """
    filename = pickle_name + ".pickle"
    print("...saving pickle")
    tmp = open(filename, 'wb')
    pickle.dump(data, tmp)
    print("pickle saved")


def run_all(filepath, collection, start, end, segment_num=20):
    """
    Runs the entire script taking in a ID_list and from the segment list to go
    over. This was done since there are API limitations and I did not want to
    go over the limits. Furthermore every now and then the API times out so I
    had to handle that in some fashoin.

    paramters:
    --------------------------
    filepath (string) -> The filepath of the ID list to use

    collection(MongoDB collection object) -> The collection object where you
    would like to start information

    start (int) -> THe position of the segmented concat list to start scraping
    from.

    end (int) -> The position of the segmented concat list to end scraping on.

    segment_num (int) -> The number of product to call at once from the API

    data (dict) -> A JSON dictionary of product information

    returns:
    --------------------------
    None, uploads all info into a MongoDB.
    """
    id_list = open_csv_with_ids(filepath)
    print("id_list imported successfully.")
    unique_id_list = remove_duplicates(id_list)
    print("unique list created successfully.")
    segmented_ids = segment_and_concat_id_list_for_api(unique_id_list,
                                                       segment_num)
    print("segmented list created successfully, {} total segments".format(
                                                      len(segmented_ids)))
    short_list = segmented_ids[start:end]
    for number, segment in enumerate(short_list):
        try:
            data_dict = load_json_from_api(segment, SECRET_KEY)
        except:
            # Error handles HTML timeout from walmarts gateat error 504
            time.sleep(2)
            print('Error Handled')
            try:
                data_dict = load_json_from_api(segment, SECRET_KEY)
            # If the error happens more than once on a given number it is
            # likely a 400 error for an invalid product ID, just skip it.
            except:
                print('400 error')
                continue

        for item in data_dict['items']:
            collection.replace_one(
                {'itemId': item['itemId']},
                item, upsert=True)
        print('segment complete {}/{}, there are {} items in the collection.'
              .format(number + start, len(segmented_ids), collection.count()))


if __name__ == '__main__':
    # Instantiate Mongo
    client = MongoClient()
    db = client.capstone_project
    collection = db.product_id_returns

    # Pass in command line arguments
    start = int(sys.argv[1])
    end = int(sys.argv[2])

    # Run and store in the data folder
    run_all('../../data/all_ids', collection, start, end)
