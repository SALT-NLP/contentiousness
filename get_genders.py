import pandas as pd
import ast
import sys

if len(sys.argv) != 2:
    print("Usage: get_genders.py data_file")
    sys.exit(-1)

COMPLETE_DATA = sys.argv[1] # data w/ bigrams. only used if CREATE BIGRAMS = False

data = pd.read_csv('user_information.csv', sep=';')

def to_list(x):
    return ast.literal_eval(x)
    
genders = data.genders.apply(to_list)
genders = genders.apply(lambda x: x[0] if len(x) > 0 else None)
users = data.user_name

assert(len(genders) == len(users))

gender_dict = {user: gender for user, gender in zip(users.values, genders.values)}

import pickle

complete_data = pd.read_pickle(COMPLETE_DATA)

def count_genders(thread):
    result = {
        'male': 0,
        'female': 0,
        None: 0
    }
    try:
        result[gender_dict[thread.author]] += 1
    except:
        pass
    
    for comment in thread.comments:
        try:
            result[gender_dict[comment['author']]] += 1
        except:
            continue
        
    del result[None]
    return result
    

gender_counts = complete_data.apply(count_genders, axis=1)
male_count = gender_counts.apply(lambda x: x['male'])
female_count = gender_counts.apply(lambda x: x['female'])

complete_data['male'] = male_count
complete_data['female'] = female_count

complete_data.to_pickle(COMPLETE_DATA)