import argparse
import json
import os
import re
from pathlib import Path
import curses

import jsonlines
from openai import OpenAI

import getting_user_input
from utils import (
    addstr_wordwrap,
    CursesWindow,
    curses_overflow_restarts,
    ScreenOwner,
    print_job_done,
)


class OpenAIGeneration:
    R_EXTRACT = re.compile(r"```json\n(.*)\n```", re.DOTALL)

    model = "gpt-4-turbo"

    system = "You are an expert on history, culture, law and Czech linguistics."

    prompts = {
        "alternative": """You are given an input text. Create 5 Czech descriptions of the text as a whole, each at most 4 words long.
    Each description has to be a common phrase, concrete concept or professional terminology.
    Don't make up new word combinations. Match the context and historical period of the text.

    For each description, create two alternatives, which have the following criteria:
    - They are a common phrase or concept.
    - They describe concepts similar to the original description.
    - They DON'T describe the input text as a whole.
    - They DON'T describe any specific section of the input text.
    - They belong in the same context and historical period as the original text.
    - They CAN'T be nonsensical, too broad, or irrelevant to the background of the original text.

    Output them as a JSON dictionary of arrays, where the keys are the original descriptions and the values are your alternatives.
    Output raw data without formatting.""",
        # ----------------------------------------
        "two_sec": """You are given an input text.
    Create 5 topics in Czech that satisfy these criteria:
    1. Each correctly describes some section of the input text
    2. Each INCORRECTLY describes some OTHER section of the input text
    3. Each is at most 5 words long
    4. Each is a common phrase, concept or professional terminology
    5. Each isn't an abstract concepts
    6. They are not new or unusual word combinations

    Only focus on the written text, ignore it's context, implications, or background.
    Output a correctly formatted JSON object with two keys: 'topics' and 'explanations'.
    'topics' is a list of topics.
    'explanations' is a list describing why they satisfy these criteria in Czech in at most 10 words.
    Output raw data without formatting.""",
        # ----------------------------------------
        "slightly_wrong_desc_cze": """Zde je vstupní text. Vytvoř 5 MÍRNĚ chybných popisů textu.
    Každý popis bude mít nejvýše 3 slova. Nevytvářej zjevně chybné popisy.
    Všechny popisy musí být gramaticky správné a musí být známými frázemi, koncepty nebo odbornými termíny. Nevymýšlej nová slovní spojení.
    Zobecnění správného popisu nebo nepřímý popis obsahu se nepočítají jako chybný popis - nevypisuj je.
    Výsledné popisy by měly být schopny oklamat odborníka v oboru.
    Výstupem bude správně formátovaný JSON objekt se dvěma klíči: 'descriptions' a 'explanations'.
    'descriptions' je seznam popisů.
    'explanations' je seznam vysvětlení proč každý popis splňuje tato kritéria v maximálně 8 slovech.
    Nevypisuj formátování Prism.""",
    }

    current_prompt = prompts["alternative"]

    def __init__(self, path):
        self.data_path = path
        self.data = []

        already_generated = 0
        i = 0
        with jsonlines.open(self.data_path, mode='r') as reader:
            for i, text_obj in enumerate(reader):
                self.data.append(text_obj)
                if "llm_generated_hn" in text_obj:
                    already_generated += 1

        print(
            f"There are {already_generated} already generated hard negatives"
            f" and {i - already_generated} to generate."
        )

        self.client = OpenAI()

    def spam_api(self, take, force_regenerate):
        print(f"Generating hard negatives for {take} texts.")
        generated = 0
        for text_index, text in enumerate(self.data):
            id = text["text_id"]
            if not force_regenerate and "llm_generated_hn" in text:
                continue

            if generated == take:
                break

            prompt = f"{OpenAIGeneration.current_prompt}\nVstupní text: {text['text']}"
            completion = self.client.chat.completions.create(
                model=OpenAIGeneration.model,
                messages=[
                    {"role": "system", "content": OpenAIGeneration.system},
                    {"role": "user", "content": prompt},
                ],
            )
            response = completion.choices[0].message.content
            try:
                result = json.loads(response)
            except json.JSONDecodeError:
                print(
                    f"Error: Couldn't decode JSON response for text {id}."
                    f"Raw response: '{response}'"
                )
                continue
            if OpenAIGeneration.current_prompt == OpenAIGeneration.prompts["alternative"]:
                result = [item for sublist in result.values() for item in sublist]
            self.data[text_index]["llm_generated_hn"] = result
            print(f"Generated hard negatives for text {id}.")
            generated += 1

        if generated < take:
            print(f"Hard negatives for only {generated}/{take} texts generated.")

        with jsonlines.open(self.data_path, mode='w') as writer:
            writer.write_all(self.data)


class MergeHN:
    def __init__(self, merge_from_path, merge_to_path, take_api, take_from_dataset):
        print(f"Merging hard negatives from {merge_from_path}.")
        self.data_from = json.load(open(merge_from_path, mode="r"))
        self.data_to = []
        self.merge_to_path = merge_to_path

        self.take_api = take_api
        self.take_from_dataset = take_from_dataset

        hns_total = 0
        with jsonlines.open(merge_to_path, mode='r') as reader:
            for text in reader:
                self.data_to.append(text)
                if "potential_hard_negatives" in text:
                    hns_total += 1

        print(
            f"There are {len(self.data_to) - hns_total} texts in dataset which already have hard negatives."
            f"Merging hard negatives for {hns_total} texts."
        )
        if hns_total == len(self.data_to):
            print(
                "All texts have hard negatives."
                "You can specify --force to remerge not-annotated hard negatives."
            )

    def merge(self, hn_sort_threshold, force):
        for i, text in enumerate(self.data_to):
            id = text["text_id"]
            if not force and "potential_hard_negatives" in text:
                continue

            hn_from_dataset = self.data_from[id]["potential_negatives_all"]

            # sort hard negatives by similarity or threshold
            if hn_sort_threshold is None:
                hn_from_dataset = sorted(hn_from_dataset, key=lambda x: x["similarity"])
            else:
                hn_from_dataset = sorted(
                    hn_from_dataset,
                    key=lambda x: abs(x["similarity"] - hn_sort_threshold),
                )
            hn_from_dataset = hn_from_dataset[: self.take_from_dataset]
            hn_from_dataset = [
                {"topic": hn["topic"], "type": "from_dataset"} for hn in hn_from_dataset
            ]

            # take hard negatives from API
            try:
                hn_from_api = text["llm_generated_hn"][: self.take_api]
            except KeyError:
                print(
                    f"Warning: There are no generated hard negatives to merge for text {id}."
                )
                hn_from_api = []
            hn_from_api = [{"topic": hn, "type": "generated"} for hn in hn_from_api]

            text["potential_hard_negatives"] = hn_from_dataset + hn_from_api
            print(f"Merged hard negatives for text {id}.")

        with jsonlines.open(self.merge_to_path, mode='w') as writer:
            writer.write_all(self.data_to)


class ScreenOwnerHns(ScreenOwner):
    controls_string = (
        "Press y/Y if the topic is good hard-negative, n/N if it is not.\n"
        "You can also skip this text anytime by pressing 's'.\n"
    )

    def __init__(self, crs, text, nb_left, nb_cleaned_this_session, good_topics):
        self.good_topics = good_topics
        super().__init__(crs, text, nb_left, nb_cleaned_this_session)

    def redraw(self):
        super().redraw()

        if len(self.good_topics):
            self.crs.addstr("Correct topics: \n", curses.A_BOLD)

        for good_topic in self.good_topics:
            self.crs.addstr(f"{good_topic}\n")

        self.crs.addstr("\nHard negatives: \n", curses.A_BOLD)

    def annotation_done(self, annotated_topic, hn_count):
        annotation = "✓" if annotated_topic['annotation'] else "✗"
        type_shortcuts = {
            "from_dataset": "D",
            "generated": "G",
            "rejected": "R",
        }
        hn_type = type_shortcuts[annotated_topic["type"]]
        self.crs.addstr(f"{annotation} #{hn_count} {hn_type}: {annotated_topic['topic']}\n")

    def redraw_annotated(self, annotated_topics):
        self.redraw()

        for count, annotated_topic in enumerate(annotated_topics, start=1):
            self.annotation_done(annotated_topic, count)


class HNAnnotator:
    def __init__(self, source_path, crs):
        self.source_path = source_path
        self.data = json.load(open(source_path, mode="r"))

        self.out_json_path = str(source_path).strip(".json") + "_annotated.jsonl"
        if os.path.exists(self.out_json_path):
            with jsonlines.open(self.out_json_path, mode='r') as reader:
                self.out_data = list(reader)
        else:
            self.out_data = []
        self.curses_err_count = 0
        self.crs = crs

    def annotate_text(self, screen_owner, potential_hard_negatives):
        annotated_hard_negatives = []

        for count, hard_negative in enumerate(potential_hard_negatives, start=1):
            self.crs.addstr(f"{hard_negative['topic']} \n")
            is_good_hn = getting_user_input.accept_or_reject(self.crs, "Good hard negative? [Y/n]")

            annotated_hn = {
                "topic": hard_negative["topic"],
                "type": hard_negative["type"],
                "annotation": is_good_hn,
            }
            annotated_hard_negatives.append(annotated_hn)
            screen_owner.redraw_annotated(annotated_hard_negatives)
        return annotated_hard_negatives

    def annotate_loop(self):
        number_of_texts = len(self.data)
        # annotated texts are texts with at least one annotated hard negative
        number_of_annotated_texts = sum(
            [
                any("annotation" in hn for hn in text["potential_hard_negatives"])
                or "skipped" in text
                for text in self.data.values()
            ]
        )

        self.put_introduction(number_of_texts, number_of_annotated_texts)
        quit_or_proceed = getting_user_input.quit_or_proceed(self.crs)
        if quit_or_proceed == "quit":
            return 0

        annotated_texts_session = 0
        end = False

        for text_id in self.data:
            if "skipped" in self.data[text_id]:
                continue

            skipped = False
            if any(
                    "annotation" in hn
                    for hn in self.data[text_id]["potential_hard_negatives"]
            ):
                continue
            to_annotate = number_of_texts - number_of_annotated_texts - annotated_texts_session
            text = self.data[text_id]["text"]

            good_topics = self.data[text_id]["topics"]
            screen_owner = ScreenOwnerHns(self.crs, text, to_annotate, annotated_texts_session, good_topics)

            potential_hard_negatives = self.data[text_id]["potential_hard_negatives"]
            try:
                annotated_hard_negatives = self.annotate_text(screen_owner, potential_hard_negatives)
            except getting_user_input.SkipError:
                skipped = True
                annotated_hard_negatives = []

            try:
                if not skipped:
                    annotated_hard_negatives = self.redo_if_needed(screen_owner, annotated_hard_negatives)
            except getting_user_input.QuitError:
                end = True
            except getting_user_input.SkipError:
                skipped = True

            if skipped:
                self.data[text_id]["skipped"] = True
            else:
                selected_hns = []
                for hn in annotated_hard_negatives:
                    if hn["annotation"]:
                        selected_hns.append(hn["topic"])
                self.data[text_id]["potential_hard_negatives"] = annotated_hard_negatives
                self.out_data.append(
                    {
                        "text_id": text_id,
                        "text": text,
                        "topics": self.data[text_id]["topics"],
                        "hard_negatives": selected_hns,
                    }
                )

            # save data to input file for annotation control
            json.dump(
                self.data,
                open(self.source_path, "w"),
                indent=4,
                ensure_ascii=False,
            )

            # save data to output file for final merging
            with jsonlines.open(self.out_json_path, mode='w') as writer:
                writer.write_all(self.out_data)

            annotated_texts_session += 1

            self.crs.clear()
            self.crs.refresh()

            if end:
                break

        if number_of_texts - number_of_annotated_texts - annotated_texts_session == 0:
            print_job_done(self.crs)

    def put_introduction(self, number_of_texts, number_of_annotated_texts):
        self.crs.addstr("*******************************************\n")
        self.crs.addstr("* Welcome to the hard negative annotator! *\n")
        self.crs.addstr("*******************************************\n\n\n")

        self.crs.addstr("Statistics:\n", curses.A_BOLD)
        self.crs.addstr(f"Number of texts: {number_of_texts}\n")
        self.crs.addstr(f"Number of annotated texts: {number_of_annotated_texts}\n")
        self.crs.addstr(
            f"Number of texts left: {number_of_texts - number_of_annotated_texts}\n\n\n"
        )

        self.crs.addstr("Instructions:\n\n", curses.A_BOLD)
        self.crs.addstr(
            "You will be presented with texts and potential hard negatives for each text.\n"
            "Hard negatives from dataset are marked with 'D', generated hard negatives with 'G' and "
            "rejected hard negatives with 'R'.\n"
        )
        self.crs.addstr(
            "For each potential hard negative, you will be prompted to mark it as relevant or not.\n"
        )
        self.crs.addstr("Press y/Y if the topic is relevant, n/N if it is not.\n"
                        "You can also skip text anytime by pressing 's'.\n")

        self.crs.addstr("Your annotations will be saved after each text.\n\n\n")
        self.crs.addstr("If you want to start annotating, press 'c' or 'q' to quit.\n\n")

    def redo_if_needed(self, screen_owner, annotated_hard_negatives):
        while True:
            addstr_wordwrap(
                self.crs,
                "\n\nPress 'c' to continue, 'r' to redo if you made a mistake, 'q' to quit. ",
                0,
            )
            action = getting_user_input.redo_or_proceed(self.crs)
            if action == "redo":
                self.crs.addstr(
                    "\nToggle annotation result by pressing the number of the annotation: "
                )
                key = chr(self.crs.getch())

                while True:
                    if key.isnumeric() and 1 <= int(key) <= len(
                        annotated_hard_negatives
                    ):
                        hn_id = int(key) - 1
                        toggle_to = not annotated_hard_negatives[hn_id]["annotation"]
                        annotated_hard_negatives[hn_id]["annotation"] = toggle_to
                        screen_owner.redraw_annotated(annotated_hard_negatives)
                        self.crs.addstr(f"\nAnnotation #{key} toggled.")
                        break
                    key = chr(self.crs.getch())

            elif action == "continue":
                break
            else:
                action = getting_user_input.redo_or_proceed(self.crs)

        return annotated_hard_negatives


@curses_overflow_restarts
def run_annotation(annotation_source):
    with CursesWindow() as crs:
        annotator = HNAnnotator(annotation_source, crs)
        annotator.annotate_loop()


if "__main__" == __name__:
    # argparse to merge two json files
    # argparse option to specify how many hard negatives to take in sum
    # argparse to start annotating hard negatives
    parser = argparse.ArgumentParser(
        description="Tool for texts hard negatives generation, "
                    "finding in dataset and annotating."
    )
    # argparse to create hard negatives - options [generate, find, both]
    parser.add_argument(
        "action",
        type=str,
        choices=["generate", "merge", "annotate"],
        help="Generate hard negatives for texts"
             "'generate' generates HN using OpenAI API."
             "'merge' merges hard negatives in dataset (requires --json "
             "and --merge-max argument).",
    )

    parser.add_argument(
        "--annotate-source",
        type=Path,
        help="Source file to clean dataset. "
             "Default is data/clean_dataset.json."
             "Has to be json file with texts and potential hard negatives "
             "to select good hard negatives from.",
        default="data/clean_dataset.json",
    )

    parser.add_argument(
        "--source",
        type=Path,
        help="Source file to jsonlines with already scored topics. "
             "Default is evaluation-data/out-mlm-mpnet-base-v2-all-texts_example.jsonl."
             "Has to be jsonlines file with texts to generate hard negatives for.",
        default="evaluation-data/out-mlm-mpnet-base-v2-all-texts_example.jsonl",
    )
    parser.add_argument(
        "--take",
        dest="take",
        type=int,
        default=10,
        help="Number of texts to generate hard negatives for."
             "Use with 'generate' action to limit number of API calls.",
    )

    parser.add_argument(
        "--merge-json",
        type=Path,
        default=None,
        help="Path to json file with hard negatives to merge with clean dataset."
             "Should be json generated with `negative-exclusive-sets.py` with "
             "added similarity scores using similarity modeling.",
    )
    parser.add_argument(
        "--hn-from-api",
        type=int,
        default=3,
        help="Number of hard negatives to take from API.",
    )
    parser.add_argument(
        "--hn-from-dataset",
        type=int,
        default=2,
        help="Number of hard negatives to take from dataset specified by --merge-json."
             "Takes hard negatives closest to threshold --hn-from-dataset-threshold is specified."
             "Otherwise takes hard negatives with highest similarity score.",
    )
    parser.add_argument(
        "--hn-from-dataset-threshold",
        type=float,
        default=None,
        help="Threshold for taking hard negatives from dataset.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="'generate': Turn off skipping texts with already generated hard negatives."
             "'merge': Turn off skipping texts with already merged hard negatives.",
    )

    args = parser.parse_args()

    if args.action == "generate":
        print(f"Generating hard negatives for {args.source}.")
        src_path = Path(args.source)
        if not src_path.exists():
            print(f"File {args.source} not found.")
            exit(-1)

        print("Calling OpenAI API to generate hard negatives.")
        OpenAIGeneration(src_path).spam_api(args.take, args.force)

    if args.action == "merge":
        if args.merge_json is None:
            print(f"You must to specify --merge-json argument with 'merge' action.")
            exit(-1)

        if not args.merge_json.exists():
            print(f"File {args.json} not found.")
            exit(-1)

        merger = MergeHN(
            args.merge_json, args.source, args.hn_from_api, args.hn_from_dataset
        )
        merger.merge(args.hn_from_dataset_threshold, args.force)

    if args.action == "annotate":
        run_annotation(args.annotate_source)
