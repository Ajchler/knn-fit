from annotator_api import Annotator_API

data = {
    "annotation_task_id": "947d8ee7-38ed-49c1-87e8-0e5ecba9a482",
    "from_date": "2024-02-06",
    "to_date": "2024-02-29",
}

if __name__ == '__main__':
    api = Annotator_API('https://anotator.semant.cz', 'username', 'password')
    with api.API_session():
        response = api.post(api.base_url + '/api/task/results', data=data)
        print(response.json())

