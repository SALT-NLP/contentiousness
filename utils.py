import tqdm
import numpy as np
import pandas as pd

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer 
from transformers import BertTokenizer
from scipy.special import softmax
import torch
import pickle

from torch.utils.data import TensorDataset, DataLoader
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup

from sklearn.model_selection import train_test_split


with open('models/discourse_model.pickle', 'rb') as f:
    model = pickle.load(f)

model.eval()

import pickle

import torch

# If there's a GPU available...
if torch.cuda.is_available():    
    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


tqdm.tqdm.pandas()

lemmatizer = WordNetLemmatizer() 

def flatten_thread_r(thread_id, comments):
    extracted_comments = []
    if len(comments) == 0:
        return extracted_comments
    else:
        
        for comment in comments:
            comment_dict = {
                'id': comment['id'],
                'parent_id': comment['parent_id'] or thread_id,
                'controversiality': comment['controversiality'],
                'body': comment['body'],
                'created_utc': comment['created_utc'],
                'author': comment['author']
            }
            
            extracted_comments += [comment_dict] + flatten_thread_r(thread_id, comment['children'])
            
    return extracted_comments

def flatten_thread(thread):
    flattened_comments = flatten_thread_r(thread['id'], thread['children'])
    return flattened_comments

def label(threads):
    def percent_upvoted(x):
        if x.ups + x.downs == 0:
            return 0
        return x.ups / (x.ups + x.downs)


    threads['percent_upvoted'] = threads.apply(percent_upvoted, axis=1)
    bottom_quartile = np.percentile(threads.percent_upvoted.values, 25)
    top_quartile = np.percentile(threads.percent_upvoted.values, 75)

    print('Bottom quartile: ', bottom_quartile)
    print('Top quartile: ', top_quartile)

    def controversiality(x):
        if x <= bottom_quartile:
            return int(True)
        elif x >= top_quartile:
            return int(False)
        else:
            return -1

    threads['label'] = threads.percent_upvoted.apply(controversiality)
    threads = threads[threads.label != -1]

    return threads


def prep_data(threads):
    threads = label(threads)
    threads['comments'] = threads.progress_apply(flatten_thread, axis=1)

    return threads

def get_corpus(threads):
    corpus = []
    
    def _get_corpus(thread):
        corpus.append(' '.join([lemmatizer.lemmatize(word.lower()) for word in word_tokenize(thread.selftext)]))
        for comment in thread.comments:
            corpus.append(' '.join([lemmatizer.lemmatize(word.lower()) for word in word_tokenize(comment['body'])]))
            
    threads.progress_apply(_get_corpus, axis=1)
    
    return corpus

def get_ids(threads):
    ids = {}
    def _get_ids(thread):
        ids[thread.id] = [comment['id'] for comment in thread.comments]

    threads.progress_apply(_get_ids, axis=1)

    return ids

def _text_to_bert(threads):
    def tokenize_fn(x):
        tokenized = ['[CLS]'] + bert_tokenizer.tokenize(x)[:510] + ['[SEP]']
        
        if len(tokenized) < 512:
            tokenized += ['[PAD]'] * (512 - len(tokenized))
        tokenized = bert_tokenizer.convert_tokens_to_ids(tokenized)
        return tokenized
    
    def tokenize_comments(x):
        return [tokenize_fn(comment['body']) for comment in x]
            
    
    title_body = threads.title + ' ' + threads.selftext
    title_body = title_body.apply(tokenize_fn)
    comments = df.comments.progress_apply(tokenize_comments)
    
    return (title_body, comments)

def _create_tb_mask(x):
    return [token_id > 0 for token_id in x]

def _create_comment_mask(x):
    c = []
    for comment in x:
        c.append([token_id > 0 for token_id in comment])
    return torch.LongTensor(np.stack(c)) if len(c) > 0 else torch.LongTensor([[0] * 512])


def get_discourse_acts(threads):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    with open('discourse_model.pickle', 'rb') as f:
        discourse_model = pickle.load(f)

    title_body, comments = _text_to_bert(threads)
    title_body_mask = title_body.apply(_create_tb_mask)
    comment_mask = comments.apply(_create_comment_mask)

    title_body_tensor = torch.LongTensor(np.stack(title_body.values))
    title_body_mask = torch.LongTensor(np.stack(title_body_mask.values))

    comment_tensor = comments.progress_apply(lambda x: torch.LongTensor(np.stack(x)) if len(x) > 0 else torch.LongTensor(np.stack([[0] * 512])))
    comment_mask = torch.LongTensor(np.stack(comment_mask.values))

def _find_post(id, comments):
    for comment in comments:
        if comment['id'] == id:
            return comment
    return None

def _create_bigrams(thread):
    bigrams = []
    for comment in thread.comments:
        parent = _find_post(comment['parent_id'], thread.comments)
        if parent is None:
            bigrams.append((thread.discourse_act, comment['discourse_act']))
        else:
            bigrams.append((parent['discourse_act'], comment['discourse_act']))
            
    return bigrams

def create_bigrams(threads):
    threads['bigrams'] = threads.apply(_create_bigrams, axis=1)

def logits_to_act(logits):
    return np.argmax(softmax(logits, axis=-1), axis=-1)


def assign_acts(tb_acts, c_acts, threads):
    acts = logits_to_act(np.stack(tb_acts))
    threads['discourse_act'] = acts
    i = 0
    for idx, row in tqdm.tqdm(threads.iterrows(), total=len(threads)):
        cs = logits_to_act(c_acts[i])
        new_comments = []
        for c, c_act in zip(row.comments, cs):
            c['discourse_act'] = c_act
            new_comments.append(c)
        i += 1
        assert all(['discourse_act' in c.keys() for c in new_comments]), 'Not every comment got an act!' 
        row.comments = new_comments

def text_to_discourse_acts(threads):
    print("Vectorizing threads...")
    title_body_tensor, title_body_mask, comment_tensor, comment_mask = get_tensors(threads)
    print("Running BERT")
    tb_acts, c_acts = get_discourse_acts(title_body_tensor, title_body_mask, comment_tensor, comment_mask)
    print("Assigning acts...")
    assign_acts(tb_acts, c_acts, threads)
    print('Done.')

def get_discourse_acts(title_body_tensor, title_body_mask, comment_tensor, comment_mask):
    tb_acts = []
    c_acts = []
    for tbt, tbm, ct, cm in tqdm.tqdm(zip(title_body_tensor, title_body_mask, comment_tensor, comment_mask), total=len(title_body_tensor)):
        with torch.no_grad():
            dataset = TensorDataset(ct, cm)
            loader = DataLoader(dataset, batch_size=3)
            
            tbt = tbt.to(device)
            tbm = tbm.to(device)
            
            tb_act = model(tbt.view((1, -1)), attention_mask = tbm.view((1, -1)))[0].cpu().numpy()
            tb_acts.append(tb_act[0])
            
            cs = []
            for batch in loader:
                ct, cm = batch
                ct = ct.to(device)
                cm = cm.to(device)
                
                c_act = model(ct, cm)[0].cpu().numpy()
                for c in c_act:
                    cs.append(c)
            cs = np.stack(cs)
            c_acts.append(cs)
            
    return tb_acts, c_acts

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
def _text_to_bert(threads):
    def tokenize_fn(x):
        tokenized = ['[CLS]'] + bert_tokenizer.tokenize(x)[:510] + ['[SEP]']
        
        if len(tokenized) < 512:
            tokenized += ['[PAD]'] * (512 - len(tokenized))
        tokenized = bert_tokenizer.convert_tokens_to_ids(tokenized)
        return tokenized
    
    def tokenize_comments(x):
        return [tokenize_fn(comment['body']) for comment in x]
            
    
    title_body = threads.title + ' ' + threads.selftext
    title_body = title_body.apply(tokenize_fn)
    comments = threads.comments.progress_apply(tokenize_comments)
    
    return (title_body, comments)

def _create_tb_mask(x):
    return [int(token_id > 0) for token_id in x]

def _create_comment_mask(x):
    c = []
    for comment in x:
        c.append([int(token_id > 0) for token_id in comment])
    return torch.LongTensor(np.stack(c)) if len(c) > 0 else torch.LongTensor([[0] * 512])


def get_tensors(threads):
    title_body, comments = _text_to_bert(threads)
    title_body_mask = title_body.apply(_create_tb_mask)
    comment_mask = comments.apply(_create_comment_mask)

    title_body_tensor = torch.LongTensor(np.stack(title_body.values))
    title_body_mask = torch.LongTensor(np.stack(title_body_mask.values))

    comment_tensor = comments.progress_apply(lambda x: torch.LongTensor(np.stack(x)) if len(x) > 0 else torch.LongTensor(np.stack([[0] * 512])))
    comment_tensor = comment_tensor.values
    comment_mask = comment_mask.values
    
    return title_body_tensor, title_body_mask, comment_tensor, comment_mask

def _flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

import time
import datetime

def _format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

def finetune_bert(title_body_tensor, title_body_mask, threads):
    labels = torch.LongTensor(np.stack(threads.label.values))
    print(labels.size())
    train_tbt, test_tbt, train_tbm, test_tbm, train_labels, test_labels = train_test_split(
        title_body_tensor, title_body_mask, labels,
        test_size=0.2, random_state=42
    )
    
    training_stats = []
    
    train_dataset = TensorDataset(train_tbt, train_tbm, train_labels)
    test_dataset = TensorDataset(test_tbt, test_tbm, test_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=2)
    test_dataloader = DataLoader(test_dataset, batch_size=2)
    
    epochs = 4
    total_steps = len(train_dataloader) * epochs
    
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=2,
        output_attentions=False,
        output_hidden_states=True
    )
    
    model.cuda()    
    optimizer = AdamW(model.parameters(),
        lr=2e-5,
        eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, 
        num_warmup_steps = 0, # Default value in run_glue.py
        num_training_steps = total_steps)
    total_steps = len(train_dataloader) * epochs
    total_t0 = time.time()
    for epoch_i in range(0, epochs):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')
        model.train()
        
        t0 = time.time()
        total_train_loss = 0
        
        for step, batch in enumerate(train_dataloader):
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            
            model.zero_grad()
            
            loss, logits, _ = model(b_input_ids,
                                token_type_ids=None,
                                attention_mask=b_input_mask,
                                labels=b_labels)
            
            total_train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
        avg_train_loss = total_train_loss / len(train_dataloader)
        training_time = _format_time(time.time() - t0)
        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(training_time))
        
        print("")
        print("Running Validation...")
        t0 = time.time()
        
        model.eval()
        
        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0
        
        for batch in test_dataloader:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            
            with torch.no_grad(): 
                loss, logits, _ = model(b_input_ids,
                                      token_type_ids=None,
                                      attention_mask=b_input_mask,
                                      labels=b_labels)
                
            total_eval_loss += loss.item()
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.cpu().numpy()
            
            total_eval_accuracy += _flat_accuracy(logits, label_ids)
            
        avg_val_accuracy = total_eval_accuracy / len(test_dataloader)
        print("  Accuracy: {0:.2f}".format(avg_val_accuracy))
        
        avg_val_loss = total_eval_loss / len(test_dataloader)
        validation_time = _format_time(time.time() - t0)
        
        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))

        # Record all statistics from this epoch.
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Valid. Accur.': avg_val_accuracy,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )

    print("")
    print("Training complete!")

    print("Total training took {:} (h:mm:ss)".format(_format_time(time.time()-total_t0)))
    return model.cpu(), training_stats