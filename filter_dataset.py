import json
import os
import sys
import tarfile
from collections import defaultdict

import pandas as pd
import tqdm

"""
Filters the full_reddit dataset into one of three topics, as specified
by keywords and user input.
"""


KEY_WORDS = {
    'abortion' : {'abortion', 'pro-life', 'prolife', 'fetus', 'pro-choice', 'prochoice', 'planned parenthood', 'pro choice', 'pro life'},
    'climate_change': {'global warming', 'ice caps', 'carbon dioxide', 'coal', 'solar', 'fossil fuel', 'renewable', 'greenhouse gas', 'climate change', 'glacier'},
    'gun_control': {'ar-15', 'ar15', 'ar 15', 'nra', 'assault rifle', 'assault weapon', 'school shooting', 'background check', 'second amendment'},
}

if len(sys.argv) != 3:
    print("Usage: `filter_dataset.py topic_name full_reddit_path")
    sys.exit(-1)

if sys.argv[1] not in KEY_WORDS.keys():
    print("Topic must be one of:", list(KEY_WORDS.keys()))
    sys.exit(-1)

topic = sys.argv[1]
reddit_full_path = sys.argv[2]

topic_threads = defaultdict(list)

def has_keywords(topic, thread):
    '''Checks if a thread has any keywords in its title or body.'''
    for keyword in KEY_WORDS[topic]:
        if keyword in thread['selftext'].lower() or keyword in thread['title'].lower():
            return topic
        
    return None

def extract_threads(topic, file):
    '''Applies all filters in line with Hessel's paper and returns all threads that 
    satisfy those conditions.'''
    with open(file) as f:
        for line in tqdm.tqdm(f):
            try:
                thread = json.loads(line.rstrip())
            except:
                continue
            if thread['selftext'] == '' or thread['selftext'] == '[deleted]':
                continue
                
            if thread['ups'] + thread['downs'] == 0 or thread['ups'] / (thread['ups'] + thread['downs']) < 0.5:
                continue
                
            if not thread.get('children') or len(thread['children']) < 30:
                continue
            
            correct_topic = has_keywords(topic, thread)
            if correct_topic is not None:
                topic_threads[correct_topic].append(thread)

part_count = defaultdict(lambda: 1)     

if not os.path.exists('keyword_filtered'):
    os.mkdir('keyword_filtered')

with tarfile.open(reddit_full_path) as tar:
    for n, member in enumerate(tar.getmembers()):
        if len(member.name.split('.')) > 1 and member.name.split('.')[1] == 'jsonlist':
            print(member.name)
            tar.extract(member, '.')
            extract_threads(topic, member.name)
            os.remove(member.name)
    
        if n % 1000 == 0 and n != 0:
            df = pd.DataFrame(topic_threads[topic])
            df.to_pickle(f'keyword_filtered/{topic}_{part_count[topic]}.pickle')
            part_count[topic] += 1
            topic_threads[topic] = []

def percent_upvoted(x):
    if x.ups + x.downs == 0:
        return 0.
    return x.ups / (x.ups + x.downs)

def load_parts(topic):
    df = pd.DataFrame()
    for i in range(1, 23):
        temp_df = pd.read_pickle(f'keyword_filtered/{topic}_{i}.pickle')
        df = df.append(temp_df, ignore_index=True)
        
    return df

if not os.path.exists('keyword_data'):
    os.mkdir('keyword_data')

print(topic + ":")
threads = load_parts(topic)
threads.to_pickle(f'keyword_data/{topic}.pickle')
print(threads)
print(f'Data for topic: {topic} successfully filtered.')
