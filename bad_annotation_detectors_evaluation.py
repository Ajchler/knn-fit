import json


def print_results(name, true_positive, true_negative, false_negative, false_positive):
    print(
        f"{name} evaluation:\n"
        f"\tTrue positive: {true_positive}\n"
        f"\tTrue negative: {true_negative}\n"
        f"\tFalse negative: {false_negative}\n"
        f"\tFalse positive: {false_positive}"
    )


class Detector:
    def evaluate_annotations(self):
        raise NotImplementedError("Subclasses must implement this method")


class ModeledTopicsDetector(Detector):
    MLM_THRESHOLD = 0.4
    CE1TO1_THRESHOLD = 0.4
    CE_THRESHOLD = 0.4

    def evaluate_annotations(self, data, golden_data):
        mlm_false_positive = 0
        mlm_false_negative = 0
        mlm_true_positive = 0
        mlm_true_negative = 0
        for text in golden_data:
            ce_scores = data[text]["scoring"]["ce_scores"]
            ce_scores_1to1 = data[text]["scoring"]["ce_scores_1to1"]
            mlm_scores_1to1 = data[text]["scoring"]["mlm_scores_1to1"]

            # mlm_scores_1to1
            for t in mlm_scores_1to1:
                max_score = -1
                topic = t[0]["from"]
                label = golden_data[text][topic]
                for s in t:
                    if s["score"] > max_score:
                        max_score = s["score"]

                if max_score >= self.MLM_THRESHOLD and label == 0:
                    mlm_true_negative += 1
                elif max_score >= self.MLM_THRESHOLD and label == 1:
                    mlm_false_negative += 1
                elif max_score < self.MLM_THRESHOLD and label == 0:
                    mlm_false_positive += 1
                elif max_score < self.MLM_THRESHOLD and label == 1:
                    mlm_true_positive += 1

            # ce_scores_1to1
            for t in ce_scores_1to1:
                max_score = -1
                topic = t[0]["from"]
                label = golden_data[text][topic]
                for s in t:
                    if s["score"] > max_score:
                        max_score = s["score"]

                if max_score >= self.CE1TO1_THRESHOLD and label == 0:
                    mlm_true_negative += 1
                elif max_score >= self.CE1TO1_THRESHOLD and label == 1:
                    mlm_false_negative += 1
                elif max_score < self.CE1TO1_THRESHOLD and label == 0:
                    mlm_false_positive += 1
                elif max_score < self.CE1TO1_THRESHOLD and label == 1:
                    mlm_true_positive += 1

            for t in ce_scores:
                topic = t["to"]
                score = t["score"]
                label = golden_data[text][topic]

                if score >= self.CE_THRESHOLD and label == 0:
                    mlm_true_negative += 1
                elif score >= self.CE_THRESHOLD and label == 1:
                    mlm_false_negative += 1
                elif score < self.CE_THRESHOLD and label == 0:
                    mlm_false_positive += 1
                elif score < self.CE_THRESHOLD and label == 1:
                    mlm_true_positive += 1

        print_results(
            "MLM + Modeled Topics",
            mlm_true_positive,
            mlm_true_negative,
            mlm_false_negative,
            mlm_false_positive,
        )
        print_results(
            "CE 1to1 + Modeled Topics",
            mlm_true_positive,
            mlm_true_negative,
            mlm_false_negative,
            mlm_false_positive,
        )
        print_results(
            "CE + Modeled Topics",
            mlm_true_positive,
            mlm_true_negative,
            mlm_false_negative,
            mlm_false_positive,
        )


class MLMCosineSimilarityDetector(Detector):
    THRESHOLD = 0.4

    def evaluate_annotations(self, data):
        true_positive = 0
        true_negative = 0
        false_negative = 0
        false_positive = 0
        for text in data:
            for topic in data[text]:
                _, score, target = topic.values()
                if score >= self.THRESHOLD and target == 0:
                    true_negative += 1
                elif score >= self.THRESHOLD and target == 1:
                    false_negative += 1
                elif score < self.THRESHOLD and target == 0:
                    false_positive += 1
                elif score < self.THRESHOLD and target == 1:
                    true_positive += 1

        print_results(
            "MLM Cosine Similarity",
            true_positive,
            true_negative,
            false_negative,
            false_positive,
        )


if __name__ == "__main__":
    golden_data = json.load(open("data/gold_annotated_dataset.json", "r"))
    golden_data = {entry["text"]: entry["topics"] for entry in golden_data.values()}

    data_modeled_topics = json.load(open("evaluation-data/out-eval.json", "r"))
    data_modeled_topics = {
        item["text"]: {k: v for k, v in item.items() if k != "text"}
        for item in data_modeled_topics
    }
    modeled_detector = ModeledTopicsDetector()
    modeled_detector.evaluate_annotations(data_modeled_topics, golden_data=golden_data)

    data_mlm_cosine_similarity = json.load(
        open("evaluation-data/out_mlm_cos_similarity_scores.json", "r")
    )
    data_mlm_cosine_similarity = {
        entry["text"]: entry["scores"] for entry in data_mlm_cosine_similarity.values()
    }
    mlm_detector = MLMCosineSimilarityDetector()
    mlm_detector.evaluate_annotations(data_mlm_cosine_similarity)
