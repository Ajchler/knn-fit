import requests
from contextlib import contextmanager
import logging

class Annotator_API():
    def __init__(self, base_url: str, login: str, password: str):
        self.base_url = base_url
        self.login = login
        self.password = password
        self.session = None

    def login_token(self):
        login_request = self.session.post(self.base_url + '/api/token', data={'username': self.login, 'password': self.password}, verify=False)
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
