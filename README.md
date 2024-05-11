

# Installation
To install the required dependencies, follow these steps:

Ensure you have Python installed on your system.
```shell
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

# Data pipeline
## Scraping data
You should first scrape data from API by running:
```shell
python parse_annotations.py
```

Raw scraped API data are in `data/out.json` file. Records containing zero annotator topics were removed in `data/out-clean.json`. 

## Finding bad annotations 
### Compute similarity scores for text-topic pairs

Script `similarity_modeling.py` contains function `create_text_topics_scores()` which computes cosine similarities for text and topics pairs. We tried multiple models, therefore there are multiple outputs files: `evaluation-data/out-mlm*.json`.


### Generate relevant topics
In order to experiment with generated topics, run:
```shell
python topic_modelling.py
```
You might need to change the path to the scraped json, since by default it runs with gold dataset, which you can use to check what the script does. 

Then, to score the generated topics, run:
```shell
python evaluate_topic_modelling.py
```
The resulting json file can be found in `evaluation-data/out-eval-golden.json` for the golden dataset. Note that there are multiple metrics for each generated topic in this file.

## Hard negatives
### Generating hard negatives

You can generate hard negatives for text by running:
```shell
python generate_hard_negatives.py
```

The experiments were extensive resulting in multiple hard negative files `evalutaion-data/hard-negatives*.json`.

### Finding hard negatives in the dataset
Scripts which computes similarity scores between all texts and founds exclusive sets of topics can be run by:
```shell
python negatives-exclusive-sets.py
```

Then, to compute similarity scores between found potential hard negatives and texts, please refer to the
script `similarity_modeling.py` and function `create_hard_negatives_scores()`. 
Python notebook `analyze_hard_negatives.ipynb` is then used to sort this topics by A) the highest value B) value closest to threshold (0.4). 

Resulting json files are `evaluation-data/neg_exSets-scores-sorted.json` and  `evaluation-data/neg_exSets-scores-sorted04.json`.

### Merged hard negatives
**Finally**, to see hard negative sets from multiple sources at a single place, you can refer to `evaluation-data/merged_hard_negatives.json` 
