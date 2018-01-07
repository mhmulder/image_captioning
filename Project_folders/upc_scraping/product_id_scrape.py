import requests
from bs4 import BeautifulSoup
from urllib.request import urlopen as uReq
import numpy as np
import pandas as pd
import re
import time
import csv
import math
import glob
import os


def load_main_search(query, price_min, price_max):
    """
    Queries walmarts site and returns product ids from each pages

    paramters:
    --------------------------
    query (str) -> The search query
    price_min (int) -> The low end to of the prices to query
    price_max (int) -> The high end to of the prices to query

    returns:
    --------------------------
    ids (list) -> A list of product ids
    """

    price_sort = ['high', 'low']
    ids = []
    for i in range(price_min, price_max + 1):
        print('Current query from ${} to ${}.'.format(i, i+1))
        url = ("https://www.walmart.com/search/" +
               "?cat_id=4044_103150&grid=true&max_price=" +
               "{0}&min_price={1}&page=1&query={2}&red".format(i+1, i, query) +
               "irect=false&soft_sort=false&sort=price_high#search" +
               "ProductResult"
               )
        r = requests.get(url)
        soup = BeautifulSoup(r.content, "lxml")
        num_items_class = soup.findAll("div",
                                       {"class": "result-summary-container"})
        for elem in num_items_class:
            span = str(elem.findAll("span")[1])
            num_of_items = int("".join(re.findall(r"\d", span)))
            print("In this range there are {} items.".format(num_of_items))
            if num_of_items <= 40:
                scrape_info = scrape_information(soup)
                ids.append(scrape_info)
                time.sleep(np.random.normal(1.3, 1/5))
                print("{} id's appended from page 1".format(len(scrape_info)))
            if num_of_items > 40:
                pages = math.ceil(num_of_items/40)
                if pages > 25:
                    pages = 25
                    for price in price_sort:
                        for page in range(1, pages+1):
                            time.sleep(np.random.normal(1.3, 1/5))
                            url = ("https://www.walmart.com/search/?cat_id=" +
                                   "4044_103150&grid=true&max_" +
                                   "price={0}&min_price={1}&page={2}"
                                   .format(i+1, i, page) +
                                   "&query={0}&redir".format(query) +
                                   "ect=false&soft_sort=false&sort=" +
                                   "price_{0}#searchProductResult"
                                   .format(price))
                            r = requests.get(url)
                            soup = BeautifulSoup(r.content, "lxml")
                            scrape_info = scrape_information(soup)
                            ids.append(scrape_info)
                            print("{0} id's appended from page {1}/{2}"
                                  .format(len(scrape_info), page, pages) +
                                  " using the {0} sort.".format(price))
                else:
                    for page in range(1, pages+1):
                        time.sleep(np.random.normal(1.3, 1/5))
                        url = ("https://www.walmart.com/search/?cat_id=4044_" +
                               "103150&grid=true&max_price=" +
                               "{0}&min_price={1}&page={2}&query={3}"
                               .format(i+1, i, page, query) +
                               "&redirect=false&soft_sort=false&sort=price_" +
                               "high#searchProductResult")
                        r = requests.get(url)
                        soup = BeautifulSoup(r.content, "lxml")
                        scrape_info = scrape_information(soup)
                        ids.append(scrape_info)
                        print("{} id's appended from page {}/{}."
                              .format(len(scrape_info), page, pages))
    return ids


def scrape_information(soup):
    """
    Scrape id from a beautiful soup objects

    paramters:
    --------------------------
    soup (bs4 soup object) -> lxml soup objects

    returns:
    --------------------------
    ids (list) -> A list of product ids
    """
    id_data = soup.findAll("div",
                           {"class": "display-inline-block " +
                            "pull-left prod-ProductCard--Image"})
    product_id_re = re.compile("/\d{7,10}")
    soup_string = str(id_data)
    ids = re.findall(product_id_re, soup_string)
    return ids


def run(query, price_min, price_max, filename):
    """
    Runs the scraping script and takes a query and price ranges into multiple
    csvs of product ids.

    paramters:
    --------------------------
    query (str) -> The search query
    price_min (int) -> The low end to of the prices to query
    price_max (int) -> The high end to of the prices to query
    filename (str) -> Designated filename for the unique id lists

    returns:
    --------------------------
    None, but creates numerous csv's in the parent folder.
    """
    price_range_by_ten = np.arange(price_min, price_max, 10)
    for i in range(0, len(price_range_by_ten)-2):
        print('Now starting range {} to {}'.format(price_range_by_ten[i],
                                                   price_range_by_ten[i+1]))
        ids = load_main_search(query, price_range_by_ten[i],
                               price_range_by_ten[i+1])
        unique_ids = clean_id_list(ids)
        print("Saving {} unique ids".format(len(unique_ids)))
        save_2_csv(unique_ids, str(filename +
                                   str(price_range_by_ten[i]) +
                                   str(price_range_by_ten[i+1])))


def clean_id_list(id_list):
    """
    Takes an idlist generated by the scrape_information function and returns a
    list of clean and unique ids.

    paramters:
    --------------------------
    id_list(list) -> A list of unclean id's generated from the
                     scrape_information function

    returns:
    --------------------------
    unique_clean_id_list(list) -> A list of unique clean id ints
    """
    clean_id_list = []
    for elem in id_list:
        for nested_elem in elem:
            clean_id_list.append(int(nested_elem[1::]))
    unique_clean_id_list = list(set(clean_id_list))
    return unique_clean_id_list


def save_2_csv(clean_ids, filename):
    """
    Takes a list of clean_ids and creates a csv of them for later use.

    paramters:
    --------------------------
    clean_ids(list) -> A list of clean unique id's
    filename(str) -> The filename for the csv

    returns:
    --------------------------
    None, but creates a csv in the parent directory
    """
    with open(filename, 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(clean_ids)
    print("File saved as {}".format(filename))


def concat_all_csv(output_filename):
    """
    Goes into the data folder within the current directory and scrapes
    all files (glob) than opens and concatenates all id's from all folders.
    Creates one total csv with all info.

    paramters:
    --------------------------
    output_filename (str) -> the name of the output file

    returns:
    --------------------------
    None, but creates a csv in the parent directory
    """
    files = glob.glob(os.getcwd() + "/data/*")
    all_ids = []
    for fle in files:
        with open(fle, 'r') as mycsv:
            ids = csv.reader(mycsv, delimiter=',')
            id_list = [id_num for id_num in ids][0]
            all_ids.extend(id_list)
    no_dupes_id_list = list(set(all_ids))
    save_2_csv(no_dupes_id_list, output_filename)


if __name__ == '__main__':
    # start the query for furniture
    idlist = run("furniture", 10, 511, 'data_day3/unique_ids_day_3')
