from controversy_pipeline import ContentiousVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import sys
import pandas as pd
import numpy as np

vectorizer = ContentiousVectorizer()
scaler = StandardScaler()
model = LogisticRegression(solver='liblinear', multi_class='auto', max_iter=1000)

pipeline = Pipeline([("vectorizer", vectorizer), ("scaler", scaler), ("model", model)])

X = pd.read_pickle(sys.argv[1])
y = X.label

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

pipeline.fit(X_train, y_train)
print(model.coef_.shape)
print(vectorizer.get_labels())

coef_df = pd.DataFrame()
coef_df["label"] = vectorizer.get_labels()
print(np.exp(model.coef_).shape)
coef_df["odds_ratio"] = np.exp(model.coef_[0])

print(coef_df)
coef_df = coef_df[coef_df.label != 'Text']
coef_df = coef_df.sort_values(by='odds_ratio', ascending=False)
coef_df.to_csv(sys.argv[2] + "_coef.csv", sep=";")