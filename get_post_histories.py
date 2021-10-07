import os
import pickle
import pprint
import sys
from collections import defaultdict

import pandas as pd
import praw

# The info here must be replacedw with your own Reddit API info.
reddit = praw.Reddit(client_id='REPLACE ME', client_secret='REPLACE ME', user_agent='REPLACE ME')

if len(sys.argv) != 2:
    print("Usage: get_post_histories.py data_file")
    sys.exit(-1)

df = pd.read_pickle(sys.argv[1])
comments = defaultdict(list)

users = []

def get_users(thread):
    users.append(thread.author)
    for comment in thread.comments:
        users.append(comment['author'])
        
df.apply(get_users, axis=1)
users = set(users)
print(users)

if not os.path.exists('accounts'):
    os.mkdir('accounts')

existing_accounts = os.listdir('accounts')

for user in users:
    posts = []
    if user + '.pickle' in existing_accounts:
        print("Skipping ", user)
        continue
    u = reddit.redditor(user)
    i = 0
    try:
        for comment in u.comments.top():
            if i > 100:
                break
            posts.append(comment)
            i += 1
    except:
        continue
        
    with open('accounts/' + user + '.pickle', 'wb') as f:
        pickle.dump(posts, f)

for path in os.listdir('accounts/'):
    user = path.split('.')[0]
    path = os.path.join('accounts', path)
    if os.path.exists(f'posts/{user}.pickle'):
        print("Skipping")
        continue
    try:
        with open(path, 'rb') as f:
            post_ids = pickle.load(f)
    except:
        print('continuing')
        continue
    posts = []
    i = 0
    for post_id in post_ids:
        if i == 100:
            continue
        i += 1
        try:
            sub = reddit.comment(id=post_id)
            posts.append(sub.body)
        except:
            continue
            
    with open(f'posts/{user}.pickle', 'wb') as f:
        pickle.dump(posts, f)

users = []

for path in os.listdir('posts'):
    user = path.split('.')[0]
    path = os.path.join('posts', path)
    try:
        with open(path, 'rb') as f:
            user_dict = {
                'user_name' : user,
                'description': '',
                'comments': pickle.load(f)
            }

        users.append(user_dict)
    except:
        print("skipping, invalid")
    
import json
with open('outputfile.json', 'w') as f:
    json.dump(users, f)