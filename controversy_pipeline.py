import argparse
import os
import sys
import itertools

import pandas as pd
from scipy.stats import entropy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state
from sklearn.preprocessing import scale
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import Normalizer
from sklearn.neural_network import MLPClassifier

from utils import *

bert = None
discourse_acts = ['question', 'answer', 'announcement', 'agreement', 'appreciation', 'disagreement', 'negativereaction', 'elaboration', 'humor', 'other']
possible_acts = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
possible_bigrams = itertools.product(possible_acts, possible_acts)
possible_bigrams = list(possible_bigrams)

class ContentiousVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, random_state=None, **kwargs):
        self.random_state = random_state
        self.tfidf_vectorizer = TfidfVectorizer(min_df=1, max_df=0.95, max_features=1000, stop_words='english')
        self.ohe = OneHotEncoder(handle_unknown="ignore")

    def fit(self, X, y, **kwargs):
        self.random_state_ = check_random_state(self.random_state)
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a dataframe.")
        if "body" not in X.columns and "comments" not in X.columns:
            raise ValueError("X must contain body and comments columns.")

        corpus = get_corpus(X)
        self.tfidf_vectorizer.fit(corpus)
        self.ohe.fit(X.subreddit.values.reshape((-1, 1)))
        
        self.locations = set()
        for location_list in X.locations:
            for location in location_list:
                self.locations.add(location)
        self.locations = sorted(list(self.locations))
        self.coded_locations = set()
        for coded_location_list in X.coded_locations:
            for coded_location in coded_location_list:
                self.coded_locations.add(coded_location)
        self.coded_locations = sorted(list(self.coded_locations))

        self.comment_liwc = set()
        for liwc_dict in X.comment_liwc:
            for key in liwc_dict:
                self.comment_liwc.add(key)
        self.comment_liwc = sorted(list(self.comment_liwc))

        self.selftext_liwc = set()
        for liwc_dict in X.selftext_liwc:
            for key in liwc_dict:
                self.selftext_liwc.add(key)
        self.selftext_liwc = sorted(list(self.selftext_liwc))

        return self

    def _vectorize_threads(self, thread, **kwargs):
        vectorized_thread = self.tfidf_vectorizer.transform([thread.selftext]).toarray()[0]
        vectorized_comments = self.tfidf_vectorizer.transform([comment['body'] for comment in thread.comments]).toarray()
        vectorized_comments = vectorized_comments.mean(axis=0)

        return vectorized_thread + vectorized_comments

    def _vectorize_bigrams(self, bigrams, **kwargs):
        vectorized_bigrams = np.zeros(100)
        for bigram in bigrams:
            vectorized_bigrams[possible_bigrams.index(bigram)] += 1
            
        return vectorized_bigrams

    def _vectorize_unigrams(self, thread, **kwargs):
        unigrams = np.zeros(10)
        unigrams[thread.discourse_act] += 1
        
        for comment in thread.comments:
            unigrams[comment['discourse_act']] += 1
            
        return unigrams

    def _vectorize_selftext_liwc(self, thread, **kwargs):
        liwc = np.zeros(len(self.selftext_liwc))
        for key, value in thread.selftext_liwc.items():
            liwc[self.selftext_liwc.index(key)] += value
        
        return liwc

    def _vectorize_comment_liwc(self, thread, **kwargs):
        liwc = np.zeros(len(self.comment_liwc))
        for key, value in thread.comment_liwc.items():
            liwc[self.comment_liwc.index(key)] += value
        
        return liwc

    def get_labels(self):
        text_labels = ["Text"] * self.vectorized_text.shape[1]
        discourse_labels = discourse_acts
        bigram_labels = [f"({discourse_acts[a]},{discourse_acts[b]})" for a, b in possible_bigrams]
        location_labels = self.locations
        coded_location_labels = self.coded_locations
        days_label = ["Days"]
        gender_labels = ["Male", "Female"]
        prolific_labels = ["Prolific [25]", "Prolific [50]", "Prolific [100]"]
        toxicity_labels = ["Title toxicity", "Avg. selftext toxicity", "Avg comment toxicity", "Max. selftext toxicity", "Max. comment toxicity"]
        vader_labels = ["Max. selftext VADER", "Min. selftext VADER", "Avg. selftext VADER",
                   "Max. comment VADER", "Min. comment VADER", "Avg. comment VADER"]
        liwc_labels = self.selftext_liwc + self.comment_liwc
        # subreddit_labels = self.ohe.categories_ + ["Unknown subreddit"]

        return text_labels + discourse_labels + bigram_labels + location_labels + coded_location_labels + days_label + gender_labels + prolific_labels + toxicity_labels + vader_labels + liwc_labels

    def transform(self, X, **kwargsv):
        if bert is not None:
            print("Vectorizing BERT text.")
            vectorized_text = np.stack(X.bert)
            self.vectorized_text = np.stack(X.bert)
        else:
            vectorized_text = np.stack(X.apply(self._vectorize_threads, axis=1).values)
            self.vectorized_text = np.stack(X.apply(self._vectorize_threads, axis=1).values)

        vectorized_unigrams = np.stack(X.apply(self._vectorize_unigrams, axis=1).values)
        vectorized_bigrams = np.stack(X.bigrams.apply(self._vectorize_bigrams).values)

        vectorized_locations = np.zeros((len(X), len(self.locations)))
        for n, locations_list in enumerate(X.locations):
            for location in locations_list:
                try:
                    vectorized_locations[n][self.locations.index(location)] += 1
                except:
                    continue

        for n in range(len(X)):
            for i in range(len(vectorized_locations[n])):
                if vectorized_locations[n][i] <= 3:
                    vectorized_locations[n][i] = 0

        vectorized_coded_locations = np.zeros((len(X), len(self.coded_locations)))
        for n, coded_locations_list in enumerate(X.coded_locations):
            for coded_location in coded_locations_list:
                try:
                    vectorized_coded_locations[n][self.coded_locations.index(coded_location)] += 1
                except:
                    continue

        for n in range(len(X)):
            for i in range(len(vectorized_coded_locations[n])):
                if vectorized_coded_locations[n][i] <= 25:
                    vectorized_coded_locations[n][i] = 0

        days = X.days.apply(lambda x: np.mean(x))
        days = days.values.reshape((-1, 1))
        male = X.male.values.reshape((-1, 1))
        female = X.female.values.reshape((-1, 1))
        prolific_25 = X.prolific_25.values.reshape((-1, 1))
        prolific_50 = X.prolific_50.values.reshape((-1, 1))
        prolific_100 = X.prolific_100.values.reshape((-1, 1))
        subreddit = self.ohe.transform(X.subreddit.values.reshape((-1, 1)))
        subreddit = subreddit.toarray()

        toxicity = X[["title_toxicity", "average_selftext_toxicity", "average_comment_toxicity", "max_selftext_toxicity", "max_comment_toxicity"]].values
        vader = X[["max_selftext_vader_score", "min_selftext_vader_score", "avg_selftext_vader_score",
                   "max_comment_vader_score", "min_comment_vader_score", "avg_comment_vader_score"]]

        selftext_liwc = np.stack(X.apply(self._vectorize_selftext_liwc, axis=1).values)
        comment_liwc = np.stack(X.apply(self._vectorize_comment_liwc, axis=1).values)
        new_X = np.hstack(
            (vectorized_text, 
            vectorized_unigrams, 
            vectorized_bigrams,
            vectorized_locations,
            vectorized_coded_locations,
            days,
            male,
            female,
            prolific_25,
            prolific_50,
            prolific_100,
            toxicity,
            vader,
            selftext_liwc,
            comment_liwc
            )
        )

        return new_X


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Performs data processing and runs experiments")
    parser.add_argument('--topic', action='store', default='abortion', help="The topic to analyze.")
    parser.add_argument('--data', action='store', default=None, help="The path to thhe data to process which includes all features.")
    parser.add_argument('--bert', action='store', help="The path to the computed BERT representation.")
    parser.add_argument('--fraction', action='store', default=1.0, help="The fraction of comments to use (float).", type=float)
    parser.add_argument('--outfile', action='store', default="outfile.txt", help="The name of the output file to create.")

    args = parser.parse_args()

    topic = args.topic
    fraction = args.fraction
    output_file = args.outfile

    COMPLETE_DATA = args.data
    BERT_BASELINE = args.bert

    for file in os.listdir('keyword_data'):
        key = file.split('.')[0]
        if key != topic:
            continue
        print('=== ' + key + ' ===')
        df = pd.read_pickle(os.path.join('keyword_data', file))
        threads = prep_data(df)

    def get_frac(comments):
        comments = sorted(comments, key=lambda x: int(x['created_utc']))
        frac_idx = int(len(comments) * fraction)
        return comments[:frac_idx + 1]

    threads.comments = threads.comments.progress_apply(get_frac)

    if COMPLETE_DATA:
        threads = pd.read_pickle(COMPLETE_DATA)

    if BERT_BASELINE:
        with open(BERT_BASELINE, 'rb') as f:
            bert = pickle.load(f)
            threads["bert"] = bert

    if "bigrams" not in threads.columns:
        import torch
        import pickle

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

        bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')    
        text_to_discourse_acts(threads)
        print(threads.discourse_act)
        create_bigrams(threads)
        print(threads.bigrams)
        threads.to_pickle(f"{topic}_complete.pickle")

    import pprint

    def run_experiment(name='experiment', random_state=42, C=0.01):
        y = threads.label.values
        X = threads
        vectorizer = ContentiousVectorizer()
        if bert is not None:
            # model = MLPClassifier(max_iter=10000)
            model = LogisticRegression(solver='liblinear', multi_class='auto', max_iter=1000)
        else:
            model = LogisticRegression(solver='liblinear', multi_class='auto', max_iter=1000)
        scaler = Normalizer()

        pipeline = Pipeline([
            ("vectorizer", vectorizer), 
            ("scaler", scaler), 
            ("model", model)])
        scores = cross_validate(pipeline, X, y, scoring=['f1', 'precision', 'recall', 'accuracy'])
        score_dict = {}
        for key, value in scores.items():
            score_dict[key] = value.mean()
        
        return score_dict

    print("REGULAR")
    scores = run_experiment()
    pprint.pprint(scores)


    from scipy.stats import ks_2samp

    counts_cont = threads[threads.label == 1].subreddit.sort_values().value_counts()
    counts_noncont = threads[threads.label == 0].subreddit.sort_values().value_counts()

    stat, p = ks_2samp(counts_cont, counts_noncont)
    print(p)

    acc, prec, rec, f1 = scores["test_accuracy"], scores["test_precision"], scores["test_recall"], scores["test_f1"]
    with open(f"out/{output_file}", 'w') as f:
        f.write(f"& ${round(acc, 3)}$ & ${round(prec, 3)}$ & ${round(rec, 3)}$ & ${round(f1, 3)}$")
