from sentence_transformers import SentenceTransformer
import numpy as np
import torch
from torch import nn
import json

class MLMTopicEvaluator:
    def __init__(self, *args):
        self.model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        self.cos_sim = nn.CosineSimilarity(dim=1)

    def get_embedding(self, text):
        return self.model.encode(text)

    def get_similarity(self, text, topics):
        text_embedding = self.get_embedding(text)
        topics_embeddings = self.get_embedding(topics)
        text_embedding = torch.tensor(text_embedding)
        text_embedding = text_embedding.reshape(1, -1)
        topics_embeddings = torch.tensor(np.array(topics_embeddings))
        similarities = self.cos_sim(text_embedding, topics_embeddings)
        return similarities



if __name__ == "__main__":
    evaluator = MLMTopicEvaluator()
    data = json.load(open('dataset/gold_annotated_dataset.json', 'r'))
    scores_dict = {}
    i = 0
    for d in data:
        i += 1
        text = data[d]['text']
        topics = []
        labels = []
        for t in data[d]['topics']:
            topics.append(t)
            labels.append(data[d]['topics'][t])
        scores_dict[d] = {}
        scores_dict[d]['text'] = text
        scores = []
        similarities = evaluator.get_similarity(text, topics)
        for t, s in zip(topics, similarities):
            topic_dict = {}
            topic_dict['topic'] = t
            topic_dict['similarity'] = s.item()
            topic_dict['label'] = labels[topics.index(t)]
            scores.append(topic_dict)
        scores_dict[d]['scores'] = scores

    json.dump(scores_dict, open('evaluation-data/out_mlm_cos_similarity_scores.json', 'w'), indent=4, ensure_ascii=False)

