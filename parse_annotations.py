import json5
import json
from utils import Annotator_API
import os
from dotenv import load_dotenv
load_dotenv()

data = {
    "annotation_task_id": "947d8ee7-38ed-49c1-87e8-0e5ecba9a482",
    "from_date": "2023-10-06",
    "to_date": "2024-02-29",
}

username = os.getenv('API_USER')
password = os.getenv('API_PASSWORD')
pull_annotations = True

if __name__ == "__main__":

    # Pull the data from the API
    if pull_annotations:
        api = Annotator_API('https://anotator.semant.cz', username, password)
        with api.API_session():
            results = api.post(api.base_url + '/api/task/results', data=data).json()
    else:
        with open("data.json", "r") as f:
            results = json5.load(f)

    out = {}

    for result in results:
        if result['result_type'] == 'rejected':
            continue
        result2 = json5.loads(result['result'])
        annotation_id = result['id']
        user_id = result['user_id']
        annotation_task_id = result['annotation_task_instance_id']

        res = json5.loads(result2['249a3c3d-b3f7-45ea-961e-442bcb9c85ed'])
        topics = []
        for r in res:
            topics.append(r['text'])

        with api.API_session():
            text = api.get(api.base_url + f'/api/task/task_instance/{annotation_task_id}').json()['text']

        if annotation_task_id not in out:
            out[annotation_task_id] = {
                "text": text
            }

        if annotation_id not in out[annotation_task_id]:
            out[annotation_task_id]["annotation"] = {
                "annotation_id": annotation_id,
                "user_id": user_id,
                "topics": topics
            }
        break

    with open("out-better.json", "w", encoding='utf8') as f:
        json.dump(out, f, indent=4, ensure_ascii=False)

