import pymongo
from pymongo import MongoClient
import pprint
from bson import json_util
import json
import pickle
import pprint
def load_data(filename):
    print("...loading pickle")
    tmp = open(filename,'rb')
    loadedResults = pickle.load(tmp)
    tmp.close()
    print ("results loaded")
    return loadedResults

def dump_data_into_mongodb(collection, data):
    for item in data['items']:
        collection.insert_one((item))



if __name__ == '__main__':
    data = load_data('../api_call_for_info/200_lines.pickle')
    client = MongoClient()
    db = client.capstone_project
    collection = db.product_id_returns
    db.product_id_returns.delete_many({})
    dump_data_into_mongodb(collection, data)
    results = collection.find({'$or': [{'availableOnline' : False}, {'availableOnline' : True}]},
                                    {'itemId': 1,
                                     'shortDescription' : 1,
                                     'imageEntities.mediumImage' : 1,
                                     'categoryPath' : 1,
                                     'name' : 1})
    for result in results:
        pprint.pprint(result)
    print (results.count())
