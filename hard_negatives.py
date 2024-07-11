import argparse
import json
import os
import re
from pathlib import Path
import curses

import jsonlines
from openai import OpenAI

import getting_user_input
from utils import addstr_wordwrap


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


class HNAnnotator:
    def __init__(self, source_path):
        self.source_path = source_path
        self.data = json.load(open(source_path, mode="r"))

        self.out_json_path = str(source_path).strip(".json") + "_annotated.jsonl"
        if os.path.exists(self.out_json_path):
            with jsonlines.open(self.out_json_path, mode='r') as reader:
                self.out_data = list(reader)
        else:
            self.out_data = []
        self.curses_err_count = 0
        try:
            self.crs = curses.initscr()
        except:
            print("Error initializing curses, try increasing the terminal size.")
            exit(1)

    def annotation_done(self, annotation_true, hn_type, i):
        annotation = "✓" if annotation_true else "✗"
        hn_type = "D" if hn_type == "from_dataset" else "G"
        y, x = self.crs.getyx()
        self.crs.move(y, 0)

        # Delete the last line
        self.crs.deleteln()
        # Add char at the end of last line
        self.crs.insstr(y - 1, 0, f"{annotation} #{i + 1} {hn_type}: ")
        self.crs.move(y, 0)
        self.crs.refresh()

    def confirm_skip(self):
        self.crs.addstr("\nAre you sure you want to skip this text? [Y/n] ")
        key = self.crs.getch()
        if key == ord("y") or key == ord("Y"):
            return True
        return False

    def switch_annotation(self, hn_offset, toggle_to, ins_lines_count):
        annotation = "✓" if toggle_to else "✗"
        y_old, x_old = self.crs.getyx()

        # replace first char by annotation
        self.crs.delch(y_old - hn_offset - ins_lines_count, 0)
        self.crs.insstr(y_old - hn_offset - ins_lines_count, 0, annotation)
        self.crs.move(y_old, x_old)

    def annotate(self):
        try:
            if self.curses_err_count < 100:
                self.annotate_loop()
            else:
                print(
                    "Curses error, exiting to prevent terminal corruption."
                    "Try increasing the terminal size."
                )
        except curses.error:
            self.crs.clear()
            self.crs.refresh()
            self.curses_err_count += 1
            self.annotate()

    def annotate_loop(self):
        number_of_texts = len(self.data)
        # annotated texts are texts with at least one annotated hard negative
        number_of_annotated_texts = sum(
            [
                any("annotation" in hn for hn in text["potential_hard_negatives"])
                for text in self.data.values()
            ]
        )
        crs = self.crs

        self.put_introduction(number_of_texts, number_of_annotated_texts)
        quit_or_proceed = getting_user_input.quit_or_proceed(crs)
        if quit_or_proceed == "quit":
            return 0

        annotated_texts_session = 0
        if quit_or_proceed == "proceed":
            end = False
            crs.clear()
            crs.refresh()

            for text_id in self.data:
                skipped = False
                if any(
                        "annotation" in hn
                        for hn in self.data[text_id]["potential_hard_negatives"]
                ):
                    continue

                crs.addstr("Statistics:\n", curses.A_BOLD)
                crs.addstr(f"You have annotated {annotated_texts_session} texts this session.\n")
                crs.addstr(
                    f"There are {number_of_texts - number_of_annotated_texts - annotated_texts_session} texts left.\n\n"
                )

                crs.addstr("Controls:\n", curses.A_BOLD)
                crs.addstr(
                    "Press y/Y if the topic is good hard-negative, n/N if it is not.\n"
                    "You can also skip this text anytime by pressing 's'.\n"
                )
                text = self.data[text_id]["text"]
                potential_hard_negatives = self.data[text_id][
                    "potential_hard_negatives"
                ]

                crs.addstr("\nText:\n", curses.A_BOLD)
                addstr_wordwrap(crs, text, 0)
                crs.addstr("\n")

                crs.addstr("\nCorrect topics: \n", curses.A_BOLD)
                for good_topic in self.data[text_id]["topics"]:
                    crs.addstr(f"{good_topic}\n")

                crs.addstr("\nPotential hard negatives:\n", curses.A_BOLD)
                annotated_hard_negatives = []
                count = 0
                for count, hard_negative in enumerate(potential_hard_negatives):
                    while True and not skipped:
                        crs.addstr(f"{hard_negative['topic']} \n")
                        crs.addstr("Good hard negative? [Y/n] ")
                        key = crs.getch()
                        if key == ord("s") or key == ord("S"):
                            skipped = self.confirm_skip()
                            if skipped:
                                break
                            else:
                                # remove last 3 lines (1 for prompt, 1 for topic, 1 for skip prompt)
                                y_old, x_old = self.crs.getyx()
                                crs.deleteln()
                                self.crs.move(y_old - 1, 0)
                                crs.deleteln()
                                self.crs.move(y_old - 2, 0)
                                crs.deleteln()
                                continue

                        is_good_hn = key == ord("y") or key == ord("Y")
                        annotated_hn = {
                            "topic": hard_negative["topic"],
                            "type": hard_negative["type"],
                            "annotation": is_good_hn,
                        }
                        annotated_hard_negatives.append(annotated_hn)
                        self.annotation_done(is_good_hn, hard_negative["type"], count)
                        break

                ins_lines_count = 1
                while True and not skipped:
                    crs.addstr(
                        "\n\nPress 'c' to continue, 'r' to redo if you made a mistake, 'q' to quit. "
                    )
                    ins_lines_count += 2
                    key = crs.getch()
                    if (
                            key == ord("q")
                            or key == ord("Q")
                            or key == ord("c")
                            or key == ord("C")
                            or key == ord("r")
                            or key == ord("R")
                            or key == ord("S")
                            or key == ord("s")
                    ):
                        if key == ord("q") or key == ord("Q"):
                            ins_lines_count += 1
                            crs.addstr("\nAre you sure you want to quit? [Y/n] ")
                            key = crs.getch()
                            if key == ord("y") or key == ord("Y"):
                                end = True
                                break
                        elif key == ord("r") or key == ord("R"):
                            ins_lines_count += 1
                            crs.addstr(
                                "\nToggle annotation result by pressing the number of the annotation: "
                            )
                            key = chr(crs.getch())
                            if not key.isnumeric() or int(key) not in range(1, count + 2):
                                crs.addstr("\nInvalid annotation number.")
                                ins_lines_count += 1
                                continue
                            else:
                                key = int(key)
                                toggle_to = not annotated_hard_negatives[key - 1]['annotation']
                                annotated_hard_negatives[key - 1]['annotation'] = toggle_to
                                self.switch_annotation(count - (key - 1), toggle_to, ins_lines_count)
                                crs.addstr(f"\nAnnotation #{key} toggled.")
                                ins_lines_count += 1
                        elif key == ord("c") or key == ord("C"):
                            break
                        elif key == ord("s") or key == ord("S"):
                            ins_lines_count += 1
                            if self.confirm_skip():
                                skipped = True
                                break

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

                crs.clear()
                crs.refresh()

                if end:
                    break

        crs.clear()
        curses.endwin()

    def put_introduction(self, number_of_texts, number_of_annotated_texts):
        crs = self.crs
        crs.addstr("*******************************************\n")
        crs.addstr("* Welcome to the hard negative annotator! *\n")
        crs.addstr("*******************************************\n\n\n")

        crs.addstr("Statistics:\n", curses.A_BOLD)
        crs.addstr(f"Number of texts: {number_of_texts}\n")
        crs.addstr(f"Number of annotated texts: {number_of_annotated_texts}\n")
        crs.addstr(
            f"Number of texts left: {number_of_texts - number_of_annotated_texts}\n\n\n"
        )

        crs.addstr("Instructions:\n\n", curses.A_BOLD)
        crs.addstr(
            "You will be presented with texts and potential hard negatives for each text.\n"
            "Hard negatives from dataset are marked with 'D', generated hard negatives with 'G'.\n"
        )
        crs.addstr(
            "For each potential hard negative, you will be prompted to mark it as relevant or not.\n"
        )
        crs.addstr("Press y/Y if the topic is relevant, n/N if it is not.\n"
                   "You can also skip text anytime by pressing 's'.\n")

        crs.addstr("Your annotations will be saved after each text.\n\n\n")
        crs.addstr("If you want to start annotating, press 'c' or 'q' to quit.\n\n")


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
        HNAnnotator(args.annotate_source).annotate()
