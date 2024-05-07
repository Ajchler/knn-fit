import logging
from contextlib import contextmanager
from itertools import islice
from datetime import datetime

import requests
import json
import curses


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
            topics = value['topics']
            if topics is not None:
                yield value['text'], topics, key


class Annotator_API():
    def __init__(self, base_url: str, login: str, password: str):
        self.base_url = base_url
        self.login = login
        self.password = password
        self.session = None

    def login_token(self):
        login_request = self.session.post(self.base_url + '/api/token',
                                          data={'username': self.login, 'password': self.password}, verify=False)
        if login_request.status_code != 200:
            logging.error(f'Failed to login: {login_request.status_code} {login_request.text}')
            raise Exception(f'Failed to login: {login_request.status_code} {login_request.text}')

    @contextmanager
    def API_session(self):
        try:
            self.session = requests.Session()
            self.login_token()
            yield self
        finally:
            if self.session:
                self.session.close()

    def get(self, url: str):
        response = self.session.get(url)
        if response.status_code == 401:
            self.session.post(self.base_url + '/api/token/renew')
            if response.status_code != 200:
                raise Exception(f'Failed to renew token: {response.status_code} {response.text}')
            response = self.session.get(url)
        return response

    def post(self, url: str, data: dict):
        response = self.session.post(url, json=data, verify=False)
        if response.status_code == 401:
            self.session.post(self.base_url + '/api/token/renew')
            if response.status_code != 200:
                raise Exception(f'Failed to renew token: {response.status_code} {response.text}')
            response = self.session.post(url, json=data)
        return response

class GoldDatasetCreator:
    def __call__(self, *args, **kwargs):

        with open('dataset/out.json', 'r') as f:
            data = json.load(f)

        crs = curses.initscr()

        annotated_data = {}
        counter = 0

        while True:
            crs.addstr(f"Currently there are {len(annotated_data)} texts annotated.\n")
            crs.addstr('If you want to go through the texts again, press "c", to quit pres "q".\n')
            key = crs.getch()
            if key == 113:
                break
            elif key == 99:
                crs.clear()
                crs.refresh()

            text_counter = 1
            total_texts = len(data)

            for d in data:
                crs.addstr(f"Text {text_counter}/{total_texts}\n")
                text_counter += 1
                text_topics = []
                reviewed_topics = []
                for key in data[d]:
                    if key == 'text':
                        text = data[d][key]
                    else:
                        for t in data[d][key]['topics']:
                            if t not in text_topics and t != '':
                                text_topics.append(t)

                if len(text_topics) == 0:
                    continue

                if text in annotated_data:
                    continue

                crs.addstr(f"Currently there are {len(annotated_data)} texts annotated.\n")

                crs.addstr('Text:\n')
                crs.addstr(text + '\n\n')
                crs.addstr('Topics:\n')
                for t in text_topics:
                    crs.addstr(t + '\n')
                crs.addstr('\n')

                crs.addstr('If you want to review the next text, press "n", to review press "c" or "q" to quit.\n')
                crs.refresh()

                key = crs.getch()
                if key == 110:
                    crs.clear()
                    continue
                elif key == 113:
                    break

                crs.addstr('\nFor each topic, press "0" if the topic is relevant to the text, or "1" if it is not.\n\n')
                for t in text_topics:
                    crs.addstr('\n' + t + '\n')
                    crs.refresh()
                    while True:
                        key = crs.getch()
                        if key == 48:
                            reviewed_topics.append((t, 0))
                            break
                        elif key == 49:
                            reviewed_topics.append((t, 1))
                            break
                        else:
                            continue

                crs.clear()
                crs.refresh()
                topics_dict = {t: r for t, r in reviewed_topics}
                annotated_data[d] = {
                    'text': text,
                    'topics': topics_dict
                }

        crs.clear()
        curses.endwin()
        json.dump(annotated_data, open('old-jsons/annotated_dataset.json', 'w'), indent=4, ensure_ascii=False)


class GoldDatasetConvertor:
    def __call__(self, *args, **kwargs):
        new_dataset = []
        with open("topic-generation-logs/2024-05-08_00-41-35-generated-topics.json", mode="r") as golden_dataset:
            data = json.load(golden_dataset)
            # text_topics contains generated and annotator topics for one text
            for item in data:
                item['annotator_topics'] = [t for t in item['annotator_topics'].keys()]
                new_dataset.append(item)
        json.dump(new_dataset, open('topic-generation-logs/2024-05-08_00-41-35-generated-topics.json', mode="w"),
                  indent=4, ensure_ascii=False)


if __name__ == '__main__':
    pass
    # GoldDatasetConvertor()()
    # creator = GoldDatasetCreator()
    # creator()

