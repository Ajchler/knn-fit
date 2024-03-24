import json

import numpy as np
from sentence_transformers import CrossEncoder

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
        self.ce = CrossEncoder('cross-encoder/stsb-TinyBERT-L-4')

    def calculate_matching_score(self, annotator_topics, generated_topics) -> float:
        merged_generated_topics = " ".join(generated_topics)
        result_list = self.classify_list(merged_generated_topics, annotator_topics)
        return np.mean(result_list)

    def classify_list(self, sentence, list2classify):
        pairs = list(zip([sentence] * len(list2classify), list2classify))
        return self.ce.predict(pairs).tolist()

    def result_json(self, annotator_topics, generated_topics):
        merged_generated_topics = " ".join(generated_topics)
        list_scores = self.classify_list(merged_generated_topics, annotator_topics)

        scoring_results = []
        for score, annotator_topic in zip(list_scores, annotator_topics):
            scoring_result = {
                "from": merged_generated_topics,
                "to": annotator_topic,
                "score": score
            }
            scoring_results.append(scoring_result)
        return scoring_results


if __name__ == "__main__":
    with open("2024-03-22_15-13-18-generated-topics.json", mode="r") as generated_json:
        cross_enc = CrossEncoderMetric()
        evaluator = TopicEvaluator(BasicMetric(), cross_enc)

        all_generated = json.load(generated_json)
        for generated in all_generated:
            annotator_topics = generated["annotator_topics"]
            generated_topics = generated["generated_topics"]

            ce_score = cross_enc.result_json(annotator_topics, generated_topics)
            generated["scoring"] = {
                "ce-score": ce_score
            }
        print(json.dumps(all_generated, indent=4, ensure_ascii=False))
        res = evaluator.get_results(all_generated)
        print(res)
