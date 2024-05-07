import json
import logging
from contextlib import contextmanager
from itertools import islice
from datetime import datetime

import requests


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
