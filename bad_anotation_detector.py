import json

class Detector:
    def evaluate_annotations(self):
        raise NotImplementedError("Subclasses must implement this method")

class ModeledTopicsDetector(Detector):

    def evaluate_annotations(self, data):
        for d in data:
            text = d['text']
            scores = d['scoring']
            ce_scores_1to1 = scores['ce_scores_1to1']
            ce_scores = scores['ce_scores']
            mlm_scores_1to1 = scores['mlm_scores_1to1']

            # mlm_scores_1to1
            for t in mlm_scores_1to1:
                max_score = -1
                topic = t[0]['from']
                for s in t:
                    if s['score'] > max_score:
                        max_score = s['score']

                if max_score < 0.3:
                    print(f"Possibly bad annotation detected based on mlm_scores_1to1: {topic}")

            # ce_scores_1to1
            for t in ce_scores_1to1:
                max_score = -1
                topic = t[0]['from']
                for s in t:
                    if s['score'] > max_score:
                        max_score = s['score']

                if max_score < 0.3:
                    print(f"Possibly bad annotation detected based on ce_scores_1to1: {topic}")

            # ce_scores
            for t in ce_scores:
                topic = t['from']
                if t['score'] < 0.35:
                    print(f"Possibly bad annotation detected based on ce_scores: {topic}")

class MLMCosineSimilarityDetector(Detector):

    def evaluate_annotations(self, data):
        for d in data:
            text = data[d]['text']
            for s in data[d]['scores']:
                if s['similarity'] < 0.3:
                    print(f"Possibly bad annotation detected based on mlm_cosine_similarity: {s['topic']}")


if __name__ == "__main__":
    data_modeled_topics = json.load(open('evaluation-data/out-eval.json', 'r'))
    modeled_detector = ModeledTopicsDetector()
    modeled_detector.evaluate_annotations(data_modeled_topics)

    data_mlm_cosine_similarity = json.load(open('evaluation-data/out_mlm_cos_similarity_scores.json', 'r'))
    mlm_detector = MLMCosineSimilarityDetector()
    mlm_detector.evaluate_annotations(data_mlm_cosine_similarity)

