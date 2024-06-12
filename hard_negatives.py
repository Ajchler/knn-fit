import argparse
import json
import os
import re
from pathlib import Path
import curses
from openai import OpenAI


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
        self.data = json.load(open(path, mode="r"))
        already_generated = 0
        for i, (id, text) in enumerate(self.data.items()):
            if "llm_generated_hn" in text:
                already_generated += 1

        print(
            f"There are {already_generated} already generated hard negatives"
            f" and {len(self.data) - already_generated} to generate."
        )

        self.client = OpenAI()

    def spam_api(self, take, force_regenerate):
        print(f"Generating hard negatives for {take} texts.")
        generated = 0
        for id, text in self.data.items():
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
            if (
                OpenAIGeneration.current_prompt
                == OpenAIGeneration.prompts["alternative"]
            ):
                result = [item for sublist in result.values() for item in sublist]
            self.data[id]["llm_generated_hn"] = result
            print(f"Generated hard negatives for text {id}.")
            generated += 1

        if generated < take:
            print(f"Hard negatives for only {generated}/{take} texts generated.")

        json.dump(
            self.data, open(self.data_path, mode="w"), indent=4, ensure_ascii=False
        )


class MergeHN:
    def __init__(self, merge_from_path, merge_to_path, take_api, take_from_dataset):
        print(f"Merging hard negatives from {merge_from_path}.")
        self.data_from = json.load(open(merge_from_path, mode="r"))
        self.data_to = json.load(open(merge_to_path, mode="r"))
        self.merge_to_path = merge_to_path

        self.take_api = take_api
        self.take_from_dataset = take_from_dataset

        hns_total = 0
        for id, text in self.data_to.items():
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
        for id, text in self.data_to.items():
            if not force and "potential_hard_negatives" in text:
                continue

            if "potential_hard_negatives" in text:
                is_annotated = any(
                    "annotation" in hn for hn in text["potential_hard_negatives"]
                )
                if is_annotated:
                    print(
                        f"Refused to regenerate hard negatives for text {id} "
                        f"because they are already annotated."
                    )
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

        json.dump(
            self.data_to,
            open(self.merge_to_path, mode="w"),
            indent=4,
            ensure_ascii=False,
        )


class HNAnnotator:
    def __init__(self, source_path):
        self.source_path = source_path
        self.data = json.load(open(source_path, mode="r"))
        try:
            self.crs = curses.initscr()
        except:
            print("Error initializing curses, try increasing the terminal size.")
            exit(1)

    def refresh_annotation(self, annotation_true):
        char_to_print = "✓" if annotation_true else "✗"
        y, x = self.crs.getyx()
        self.crs.move(y, 0)

        # Delete the last line
        self.crs.deleteln()
        # Add char at the end of last line
        self.crs.insstr(y - 1, 0, char_to_print + " ")
        self.crs.move(y, 0)
        self.crs.refresh()

    def annotate(self):
        number_of_texts = len(self.data)
        # annotated texts are texts with at least one annotated hard negative
        number_of_annotated_texts = sum(
            [
                any("annotation" in hn for hn in text["potential_hard_negatives"])
                for text in self.data.values()
            ]
        )
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
        crs.addstr("Press y/Y if the topic is relevant, n/N if it is not.\n")
        crs.addstr("Your annotations will be saved after each text.\n\n\n")
        crs.addstr("If you want to start annotating, press 'c' or 'q' to quit.\n\n")
        key = crs.getch()

        is_good_hn = 0
        if key == ord("c"):
            end = False
            crs.clear()
            crs.refresh()

            for text_id in self.data:
                redo = True
                while redo:
                    if any(
                        "annotation" in hn
                        for hn in self.data[text_id]["potential_hard_negatives"]
                    ):
                        redo = False
                        continue

                    crs.addstr("Statistics:\n", curses.A_BOLD)
                    crs.addstr(f"You have annotated {is_good_hn} texts this session.\n")
                    crs.addstr(
                        f"There are {number_of_texts - number_of_annotated_texts - is_good_hn} texts left.\n\n"
                    )

                    crs.addstr("Controls:\n", curses.A_BOLD)
                    crs.addstr(
                        "Press y/Y if the topic is good hard-negative, n/N if it is not.\n\n"
                    )
                    text = self.data[text_id]["text"]
                    potential_hard_negatives = self.data[text_id][
                        "potential_hard_negatives"
                    ]

                    crs.addstr("\nText:\n", curses.A_BOLD)
                    crs.addstr(text + "\n")

                    crs.addstr("\nCorrect topics: \n", curses.A_BOLD)
                    for good_topic in self.data[text_id]["topics"]:
                        crs.addstr(f"{good_topic}\n")

                    crs.addstr("\nPotential hard negatives:\n", curses.A_BOLD)
                    annotated_hard_negatives = []
                    for hard_negative in potential_hard_negatives:
                        hn_type = (
                            "D" if hard_negative["type"] == "from_dataset" else "G"
                        )
                        crs.addstr(f"{hn_type}: {hard_negative['topic']} \n")
                        crs.addstr("Good hard negative? [Y/n] ")
                        key = crs.getch()
                        is_good_hn = key == ord("y") or key == ord("Y")
                        annotated_hn = {
                            "topic": hard_negative["topic"],
                            "type": hard_negative["type"],
                            "annotation": is_good_hn,
                        }
                        annotated_hard_negatives.append(annotated_hn)
                        self.refresh_annotation(is_good_hn)

                    json.dump(
                        self.data,
                        open(self.source_path, "w"),
                        indent=4,
                        ensure_ascii=False,
                    )

                    is_good_hn += 1

                    while True:
                        crs.addstr(
                            "\n\nIf you want to continue, press 'c', to redo this text if you made a mistake press 'r', to quit press 'q'. "
                        )
                        key = crs.getch()
                        if (
                            key == ord("q")
                            or key == ord("Q")
                            or key == ord("c")
                            or key == ord("C")
                            or key == ord("r")
                            or key == ord("R")
                        ):
                            if key == ord("q") or key == ord("Q"):
                                crs.addstr("\nAre you sure you want to quit? [Y/n] ")
                                key = crs.getch()
                                if key == ord("y") or key == ord("Y"):
                                    end = True
                                    redo = False
                                    break
                            elif key == ord("r") or key == ord("R"):
                                redo = True
                                break
                            elif key == ord("c") or key == ord("C"):
                                self.data[text_id][
                                    "potential_hard_negatives"
                                ] = annotated_hard_negatives
                                redo = False
                                break

                    crs.clear()
                    crs.refresh()

                if end:
                    break

        crs.clear()
        curses.endwin()


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
        "--source",
        type=Path,
        help="Source file to clean dataset. "
        "Default is data/clean_dataset.json."
        "Has to be dictionary with keys as text ids and values as text "
        "dictionaries.",
        default="data/clean_dataset_example.json",
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
        HNAnnotator(args.source).annotate()
