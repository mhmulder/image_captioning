import requests
from bs4 import BeautifulSoup
from urllib.request import urlopen as uReq
import numpy as np
import pandas as pd
import re
import time
import csv

def load_main_search(page_number, query):
    # Get the url from the website you're looking to scrape.
    time.sleep(np.random.normal(1.3,1/5))
    url = "https://www.walmart.com/search/?page={0}&query={1}&redirect=false#searchProductResult".format(page_number, query)
    r = requests.get(url)
    soup = BeautifulSoup(r.content, "lxml")
    return soup
def run(page_range, query):
    id_list = []
    for i in range(1, page_range +1):
        print ("Now on page : {}".format(i))
        soup = load_main_search(i, query)
        new_items = scrape_information(soup)
        print ("{} items added to list".format(len(new_items)))
        id_list.extend(new_items)
    return id_list

def scrape_information(soup):
    id_data = soup.findAll("div", {"class": "display-inline-block pull-left prod-ProductCard--Image"})
    product_id_re = re.compile("/\d{7,10}")
    soup_string= str(id_data)
    ids = re.findall(product_id_re, soup_string)
    return ids

def clean_id_list(id_list):
    clean_id_list = []
    for elem in id_list:
        clean_id_list.append(elem[1::])
    return clean_id_list

def save_2_csv(clean_ids, filename):
    with open(filename, 'w', newline = '') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(clean_ids)
    print ("file saved as {}".format(filename))

if __name__ == '__main__':
    # start the query for furniture
    idlist = run(5, "furniture")
    clean_ids = clean_id_list(idlist)
    save_2_csv(clean_ids, '5_pages_product_ids')
