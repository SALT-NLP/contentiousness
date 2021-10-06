import pandas as pd
import tqdm
import praw
import sys

if len(sys.argv) != 2:
    print("Usage: get_tenures.py data_file")
    sys.exit(-1)

tqdm.tqdm.pandas()

COMPLETE_DATA = sys.argv[1]

from collections import defaultdict

df = pd.read_pickle(COMPLETE_DATA)

cakedays_dict = {}
reddit = praw.Reddit(client_id='KRfOuF8F0w-oGQ', client_secret='CWGGfwGX4zyrrJAkJWG5GeGmqMw',user_agent='tenure_grabber')

def get_cakeday(author):
    try:
        account = reddit.redditor(author)
        cakeday = account.created
        return cakeday
    except Exception as e:
        return None

def get_cakedays(thread):
    author_cakeday = get_cakeday(thread.author)
    if author_cakeday:
        cakedays_dict[thread.author] = author_cakeday
    for comment in thread.comments:
        comment_cakeday = get_cakeday(comment['author'])
        if comment_cakeday:
            cakedays_dict[comment['author']] = comment_cakeday
        
df.progress_apply(get_cakedays, axis=1)
print(cakedays_dict)

import pickle

import datetime

def calculate_tenures(thread):
    thread_creation = datetime.datetime.fromtimestamp(float(thread.created_utc))
    tenures = []
    if cakedays_dict.get(thread.author):
        thread_author_cakeday = datetime.datetime.fromtimestamp(cakedays_dict[thread.author])
        tenures.append(thread_creation - thread_author_cakeday)
    else:
        tenures.append(None)
    for comment in thread.comments:
        if cakedays_dict.get(comment['author']):
            comment_creation = datetime.datetime.fromtimestamp(float(comment['created_utc']))
            comment_author_cakeday = datetime.datetime.fromtimestamp(cakedays_dict[comment['author']])
            tenures.append(comment_creation - comment_author_cakeday)
        else:
            tenures.append(None)
        
    return tenures

tenures = df.apply(calculate_tenures, axis=1)

def split_deltas(tenures):
    ret = []
    for tenure in tenures:
        if tenure is None:
            ret.append((0, 0, 0, 0))
        else:
            ret.append((tenure.days, tenure.days/7, tenure.days/30, tenure.days/365))
            
    return ret
            
tenure_tuples = tenures.apply(split_deltas)

def calc_days(tenures):
    return [tup[0] for tup in tenures]

days = tenure_tuples.apply(calc_days)

df['days'] = days

df.to_pickle(COMPLETE_DATA)

