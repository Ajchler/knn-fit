from sentence_transformers import SentenceTransformer
import numpy as np
import torch
from torch import nn
import json
from openai import OpenAI


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
        similarities = map(lambda x: x.item, similarities)
        return similarities


class BenesEvaluator:
    def __init__(self):
        self.client = OpenAI()
        self.temperature = 0.2
        self.max_tokens = 64
        self.top_p = 1
        self.frequency_penalty = 0
        self.presence_penalty = 0

        self.system_message = (
            """
            You are Czech lingual expert with years of experience. I will give you text and a few topics and
            you will provide relevance score to a given text for each topic. 
            
            Relevance score is in range from 0 to 1. Relevance score describes 
            how much do you think that given topic is correct for that text. 
            The topic is relevant to the text only if it makes sense on its own and covers 
            substantial part of the text. That means, that even though there is some entity mentioned 
            in the text, it does not need to automatically be text topic. 
            
            Input text will be separated from topics with two new lines and each topic is on subsequent line.  
            Relevance scores will be right after the topic.
            
            Input example:
            This is a text to match with each topic.
            
            topic1
            topic2
            topic3
            
            Output example:
            topic1: 0.112
            topic2: 0.852
            topic2: 0.622
            
            """
        )

    def get_similarity(self, text, topics):
        topics = '\n'.join(topics)
        gpt4_input = f"{text}\n\n{topics}"
        print(f"Input: '{gpt4_input}'")
        generation_result = self.client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {
                    "role": "system",
                    "content": self.system_message
                },
                {
                    "role": "user",
                    "content": gpt4_input
                }
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty
        )
        generated_answer = generation_result.choices[0].message.content.split('\n')
        print(f"Output: '{generated_answer}'")
        return map(lambda x: float(x.split(': ')[1]), generated_answer)


if __name__ == "__main__":
    evaluator = BenesEvaluator()
    data = json.load(open('data/gold_annotated_dataset.json', 'r'))
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
            topic_dict['similarity'] = s
            topic_dict['label'] = labels[topics.index(t)]
            scores.append(topic_dict)
        scores_dict[d]['scores'] = scores

    json.dump(scores_dict, open('evaluation-data/out-direct-score.json', 'w'),
              indent=4, ensure_ascii=False)

