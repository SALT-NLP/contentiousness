import os
import sys

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from utils import *

TOPICS = ['climate_change', 'gun_control', 'abortion']

if sys.argv[1] not in TOPICS:
    print("Topic must be one of:", TOPICS)
    sys.exit(-1)

topic = sys.argv[1]

threads = {}

for file in os.listdir('keyword_data'):
    key = file.split('.')[0]
    if key != topic:
        continue
    df = pd.read_pickle(os.path.join('keyword_data', file))
    df = prep_data(df)
    threads = df
    
title_body_tensor, title_body_mask, _, _ = get_tensors(threads)

model, training_stats = finetune_bert(title_body_tensor, title_body_mask, threads)

baseline_fname = f"finetuned_bert_model_{topic}.pickle"

with open(baseline_fname, 'wb') as f:
    pickle.dump(model, f)

title_body_tensor, title_body_mask, comment_tensor, comment_mask = get_tensors(threads)


with open(baseline_fname, 'rb') as f:
    model = pickle.load(f)
    
model.eval()
model.cuda()

dataset = TensorDataset(title_body_tensor, title_body_mask)
loader = DataLoader(dataset, batch_size=32)

all_hidden_states_mean = []

for tbt, tbm in tqdm.tqdm(loader, total=len(loader)):
    tbt = tbt.to(device)
    tbm = tbm.to(device)
    
    with torch.no_grad():
        logits, hidden_states = model(tbt, tbm)
    
    hidden_states = hidden_states[1].cpu().numpy()
    hidden_states_mean = hidden_states.mean(axis=1)
    for hidden_state_mean in hidden_states_mean:
        all_hidden_states_mean.append(hidden_state_mean)

post_reprs = np.stack(all_hidden_states_mean)
print(post_reprs.shape)

with open(baseline_fname, 'rb') as f:
    model = pickle.load(f)
    
model.eval()
model.cuda()

comment_reprs = []

for comment_tensors, comment_masks in tqdm.tqdm(zip(comment_tensor, comment_mask), total=len(comment_tensor)):
    dataset = TensorDataset(comment_tensors, comment_masks)
    loader = DataLoader(dataset, batch_size=32)
    all_hidden_states_mean = []
    for tbt, tbm in loader:
        tbt = tbt.to(device)
        tbm = tbm.to(device)

        with torch.no_grad():
            logits, hidden_states = model(tbt, tbm)

        hidden_states = hidden_states[1].cpu().numpy()
        hidden_states_mean = hidden_states.mean(axis=1)
        for hidden_state_mean in hidden_states_mean:
            all_hidden_states_mean.append(hidden_state_mean)

            
    comment_reprs.append(np.stack(all_hidden_states_mean))

for percent in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    frac_idx = int(len(comment_reprs) * percent) + 1
    comments = []
    for comment_repr in comment_reprs:
        i = np.mean(comment_repr[:frac_idx], axis=0)
        comments.append(i)
    
    comment_repr_percent = np.stack(comments)
    assert(comment_repr_percent.shape == post_reprs.shape)
    reprs = np.hstack((post_reprs, comment_repr_percent))

    percentage = int(percent * 100)
    with open(f'{topic}_bert_repr_{percentage}_percent.pickle', 'wb') as f:
        pickle.dump(reprs, f)
