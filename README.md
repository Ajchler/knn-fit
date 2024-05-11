

# Installation
To install the required dependencies, follow these steps:

Ensure you have Python installed on your system.
```shell
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

# Data pipeline
You should first scrape data from API by running:
```shell
python parse_annotations.py
```

In order to experiment with generated topics, run:
```shell
python topic_modelling.py
```
You might need to change the path to the scraped json, since by default it runs with gold dataset, which you can use to check what the script does.


To score the generated topics, run:
```shell
python evaluate_topic_modelling.py
```

Then, you can generate topics for text by running:
```shell
python generate_hard_negatives.py
```

You can refer to `similarity_modeling.py` to create json files which computes cosine similarities for text and topics pairs.
