import json

import numpy as np
from transformers import pipeline


class TopicEvaluator:
    def __init__(self, *args):
        self.metrics = args

    def get_results(self, generated):
        results = []
        for metric in self.metrics:
            score_list = self.create_score_list(metric, generated)
            result = {
                "metric_name": metric.name,
                "score_list": score_list,
                "score_total": metric.calculate_total_score(score_list)
            }
            results.append(result)
        return json.dumps(results, indent=4, ensure_ascii=False)

    @staticmethod
    def create_score_list(metric, generated):
        scores = []
        for g in generated:
            annotator_topics = g["annotator_topics"]
            generated_topics = g["generated_topics"]
            score = metric.calculate_matching_score(annotator_topics, generated_topics)
            scores.append(score)
        return scores


class Metric:
    # Name of the metric - used in logs to see which metric calculated what numbers
    name = ""

    def calculate_matching_score(self, annotator_topics, generated_topics) -> float:
        """
        Function to calculate similarity score between annotator and generated topics
        :param annotator_topics: List of annotator created topics
        :param generated_topics: List of generated topics
        :return: Score (float) value, expressing the similarity of topics
        """
        raise NotImplementedError

    def calculate_total_score(self, score_list) -> float:
        """
        Calculates overall score for a list of scores
        :param score_list:  list of scores calculated using `calculate_matching_score` function
        :return: Returns mean value.
        """
        return np.mean(score_list)


class BasicMetric(Metric):
    name = "Simple evaluation based on word matching."

    def __init__(self):
        self.param = 4

    def calculate_matching_score(self, annotator_topics, generated_topics):
        match_scores = []
        for annotator_topic in annotator_topics:
            topic_scores = []
            for generated_topic in generated_topics:
                common_words = set(annotator_topic.split()) & set(generated_topic.split())
                topic_scores.append(len(common_words) / max(len(annotator_topic.split()), len(generated_topic.split())))
            if topic_scores:
                match_scores.append(max(topic_scores))
            else:
                match_scores.append(0)
        return np.mean(match_scores)


class CrossEncoderMetric(Metric):
    # This model was chosen because it has best score on MNLI task
    # https://www.sbert.net/docs/pretrained_cross-encoders.html#nli
    name = "cross-encoder/nli-deberta-v3-base"

    def __init__(self):
        self.classifier = None

    def calculate_matching_score(self, annotator_topics, generated_topics) -> float:
        # Load classifier to memory in the time of evaluation
        if self.classifier is None:
            self.classifier = pipeline("zero-shot-classification", model='cross-encoder/nli-deberta-v3-base')

        merged_generated_topics = " ".join(generated_topics)
        res = self.classifier(merged_generated_topics, annotator_topics)
        return np.mean(res["scores"])


if __name__ == "__main__":
    with open("2024-03-22_15-13-18-generated-topics.json", mode="r") as generated_json:
        evaluator = TopicEvaluator(BasicMetric(), CrossEncoderMetric())

        all_generated = json.load(generated_json)

        res = evaluator.get_results(all_generated)
        print(res)
