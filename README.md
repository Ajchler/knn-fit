

# Installation
To install the required dependencies, follow these steps:

Ensure you have Python installed on your system.
```shell
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

# Scraping data
You should first scrape data from API by running:
```shell
python parse_annotations.py
```

Raw scraped API data are in `data/out.json` file. Records containing zero annotator topics were removed in `data/out-clean.json`.
# Annotation dataset creation
## Finding bad annotations
There are two methods to score (text-annotation) pairs. The first method relies entirely on comparing the similarity score (cosine similarity) between the text and the annotation. The second method involves generating relevant topics using a language model (LM) and comparing them with annotator-provided topics. In the next step, these scores are used to optimize the annotation workflow.
### By computing similarity scores for text-topic pairs

Script `similarity_modeling.py` contains function `create_text_topics_scores()` which computes cosine similarities for text and topics pairs. We tried multiple models, therefore there are multiple outputs files: `evaluation-data/out-mlm*.json`.


### By generating relevant topics
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

## Adding hard negatives
Another objective is to add set of hard-negatives to each text. There are two methods to create potential hard negatives. First uses LM to generate HN to each text and the second one takes topics from most similar texts in the dataset.

To familiarize yourself with the CLI arguments run:

```shell
python hard_negatives.py --help
```



### Finding hard negatives in the dataset
Scripts which computes similarity scores between all texts and founds exclusive sets of topics can be run by:
```shell
python negatives-exclusive-sets.py
```

Then, to compute similarity scores between found potential hard negatives and texts, please refer to the
script `similarity_modeling.py` and function `create_hard_negatives_scores()`.

Resulting json files are `evaluation-data/neg_exSets-scores-sorted.json` and  `evaluation-data/neg_exSets-scores-sorted04.json`.

### Generating hard negatives

To add generated hard negatives to json file specified by `--source` (defaults to `evaluation-data/out-mlm-mpnet-base-v2-all-texts_example.jsonl`) using OpenAI API (this step is not necessary for demonstration on the example json), run:

```shell
python hard_negatives.py generate --take $NUM_OF_HARD_NEGATIVES
```

This adds `llm_generated_hn` set to each text.

### Merging hard negatives sets
To create final `potential_hard_negatives` set for each text, run:

```shell
python hard_negatives.py merge --merge-json $EXCLUSIVE_SET_HNS_WITH_SCORES --hn-from-api $NUM_HN_FROM_API --hn-from-dataset $NUM_HN_FROM_DATA
```

where `$EXCLUSIVE_SET_HNS_WITH_SCORES` could be for example `evaluation-data/neg_exSets-scores.json`. It takes `$NUM_HN_FROM_API` from `llm_generated_hn` set and `$NUM_HN_FROM_DATA` from json specified by `--merge-json` argument.

# Annotation process
## Dataset cleaning

Dataset cleaner is a TUI which makes identifying bad annotations easier, to run it make sure dependencies are installed correctly and then run:

```shell
python dataset_cleaner.py
```

Carefully read and follow instructions on the screen.

If the program crashes on launch and ends with exception from the curses library then try increasing the height of your terminal window so that the TUI can be displayed properly. This process creates a new json file with the cleaned dataset `data/clean_dataset.json`.


## Hard negatives selection

In order to start the hard negatives selection TUI you can run:
```shell
python3 hard_negatives.py annotate
```

It starts the selection for a file called `data/clean_dataset.json` and saves the results to `data/clean_dataset_annotated.jsonl`.
