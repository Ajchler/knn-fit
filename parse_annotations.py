import json5
from annotator_api import Annotator_API

data = {
    "annotation_task_id": "947d8ee7-38ed-49c1-87e8-0e5ecba9a482",
    "from_date": "2024-02-06",
    "to_date": "2024-02-29",
}

username = ""
password = ""

if __name__ == "__main__":
    with open("data.json", "r") as f:
        results = json5.load(f)

    # Pull the data from the API
    api = Annotator_API('https://anotator.semant.cz', username, password)
    with api.API_session():
        results = api.post(api.base_url + '/api/task/results', data=data).json()

    out = {}

    for result in results:
        result2 = json5.loads(result['result'])
        annotation_id = result['id']
        user_id = result['user_id']
        annotation_task_id = result['annotation_task_instance_id']

        res = json5.loads(result2['249a3c3d-b3f7-45ea-961e-442bcb9c85ed'])
        topics = []
        for r in res:
            topics.append(r['text'])

        if annotation_task_id not in out:
            out[annotation_task_id] = {}

        with api.API_session():
            text = api.get(api.base_url + f'/api/task/task_instance/{annotation_task_id}').json()

        if annotation_id not in out[annotation_task_id]:
            out[annotation_task_id][annotation_id] = {
                "user_id": user_id,
                "topics": topics,
                "text": text
            }

    with open("out.json", "w") as f:
        json5.dump(out, f, indent=4)

