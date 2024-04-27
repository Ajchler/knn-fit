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
    data = json.load(open('gold_annotated_dataset.json', 'r'))
    i = 0
    for d in data:
        i += 1
        text = data[d]['text']
        topics = []
        for t in data[d]['topics']:
            topics.append(t)

        similarities = evaluator.get_similarity(text, topics)
        print(similarities)
