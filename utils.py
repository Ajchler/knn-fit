import json
from itertools import islice
from datetime import datetime


class TopicGenerationLogger:
    def __init__(self, topic_generator, topic_evaluator, max_topic_generations, to_file=True):
        self.topic_generator = topic_generator
        self.topic_evaluator = topic_evaluator
        self.max_topic_generations = max_topic_generations
        self.generated_num = 1
        self.to_file = to_file
        time_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        self.file_logging = f"{time_string}-generation.log"
        self.file_results = f"{time_string}-generated-topics.json"

    def print_results(self, result):
        self.log("Evaluation results:")
        self.log(result)

    def print_settings(self):
        self.topic_generator.get_settings_repr()

    def new_generated(self, generated):
        if self.generated_num == 1:
            time_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.log(f"\n\nGenerating started at {time_string}")

        # Print log to stdout
        self.log(f"Generated {self.generated_num}/{self.max_topic_generations}")
        self.log(json.dumps(generated, indent=4, ensure_ascii=False))
        self.generated_num += 1

    def log(self, message):
        print(message)
        if self.to_file:
            print(message, file=open(self.file_logging, 'a'))

    def finished_generation(self, generated_all):
        with open(self.file_results, 'w') as outfile:
            json.dump(generated_all, outfile, indent=4, ensure_ascii=False)


def find_topics(item):
    for child in item.values():
        if isinstance(child, dict) and 'topics' in child:
            return [topic for topic in child['topics'] if topic != ""]
    return None


def get_annotations(file_path, num_iterations=None):
    with open(file_path, 'r') as file:
        data = json.load(file)
        for key, value in islice(data.items(), num_iterations):
            topics = find_topics(value)
            if topics is not None:
                yield value['text'], topics
