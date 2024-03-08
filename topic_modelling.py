from bertopic import BERTopic
from sklearn.datasets import fetch_20newsgroups
import json5

topic_model = BERTopic()

with open('out-non-rejected.json', 'r') as f:
    data = json5.load(f)

texts = [data[d]['text'] for d in data if d != "" ]
topics, probs = topic_model.fit_transform(texts)


