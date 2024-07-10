#!/usr/bin/env python3

import argparse
import curses
import json
import logging
import os

from utils import addstr_wordwrap

NOT_VISITED = 0
SKIPPED = 1
CHECKED = 2


controls_string = ("Press y/Y if the topic is relevant, n/N if it is not. You can also skip this text anytime by "
                   "pressing 's'.\nYou will also be able to redo the current text after last topic if you make a "
                   "mistake during cleaning.\n\n")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--INPUT_FILE', required=False, default="evaluation-data/out-mlm-mpnet-base-v2-all-texts_example.jsonl")
    parser.add_argument('--CLEAN_DATASET', default="data/clean_dataset.json")

    return parser.parse_args()


class TextDisplayer:
    def __init__(self, text, nb_left, nb_cleaned_this_session, crs):
        self.text = text
        self.nb_left = nb_left
        self.nb_cleaned_this_session = nb_cleaned_this_session

        self.crs = crs

    def display(self):
        self.crs.addstr("Statistics:\n", curses.A_BOLD)
        self.crs.addstr(f"You have cleaned {self.nb_cleaned_this_session} texts this session.\n")
        self.crs.addstr(f"There are {self.nb_left} texts left.\n\n")
        self.crs.addstr("Controls:\n", curses.A_BOLD)
        addstr_wordwrap(self.crs, controls_string, 0)
        self.crs.addstr("\nText:\n\n", curses.A_BOLD)
        addstr_wordwrap(self.crs, self.text + "\n", 0)
        self.crs.addstr("\n\n")


def display_topics(flagged_scores, current_text, crs):
    count = 0
    for score in flagged_scores:
        count += 1
        crs.addstr(f"Topic #{count}: ", curses.A_BOLD)
        crs.addstr(f"{score['topic']}\n")
        crs.addstr("Relevant? ")
        if score["topic"] in current_text["topics"]:
            crs.addstr("✓\n\n")
        else:
            crs.addstr("✗\n\n")


class CursesWindow:
    def __init__(self):
        self.crs = None

    def __enter__(self):
        try:
            self.crs = curses.initscr()
        except Exception:
            logging.error("Error initializing curses, try increasing the terminal size.")
            raise

        return self.crs

    def __exit__(self, exc_type, exc_value, traceback):
        assert self.crs is not None
        self.crs.clear()
        curses.endwin()


def put_introduction(nb_texts, nb_texts_cleaned, crs):
    crs.addstr("***********************************\n")
    crs.addstr("* Welcome to the dataset cleaner! *\n")
    crs.addstr("***********************************\n\n\n")

    crs.addstr("Statistics:\n", curses.A_BOLD)
    crs.addstr(f"Number of texts: {nb_texts}\n")
    crs.addstr(f"Number of cleaned texts: {nb_texts_cleaned}\n")
    crs.addstr(f"Number of texts left: {nb_texts - nb_texts_cleaned}\n\n\n")

    crs.addstr("Instructions:\n\n", curses.A_BOLD)
    crs.addstr("You will be presented with texts and potential topics for each text.\n")
    crs.addstr(
        "The topics are sorted by similarity to the text, with the least similar topic first.\n"
    )
    crs.addstr("For each topic you will be prompted to mark it as relevant or not.\n")
    crs.addstr("Press y/Y if the topic is relevant, n/N if it is not.\n")
    crs.addstr(
        "Once you mark a topic relevant, the rest of the topics will be marked as relevant as well.\n"
    )
    crs.addstr("After every cleaned text, your progress will be saved.\n\n\n")
    crs.addstr("If you want to start cleaning press 'c', to quit press 'q'.\n\n")


class SkipError(Exception):
    pass


class QuitError(Exception):
    pass


def user_says_accept(crs):
    crs.addstr("Relevant? [Y/n] ")

    done = False
    while not done:
        key = crs.getch()
        while key not in (ord(c) for c in "yYnNsS"):
            key = crs.getch()

        if key == ord("n") or key == ord("N"):
            return False
        elif key == ord("y") or key == ord("Y"):
            return True
        else:
            crs.addstr("\nAre you sure you want to skip this text? [Y/n] ")
            key = crs.getch()
            while key not in [ord("y"), ord("Y"), ord("n"), ord("N")]:
                key = crs.getch()

            crs.addstr("\n")
            if key == ord("y") or key == ord("Y"):
                raise SkipError


def annotate_topics(sorted_topics, crs):
    accepted_topic = False
    correct_topics = []

    for i, score in enumerate(sorted_topics, start=1):
        #  Once a topic is accepted, we accept all the following ones (based on sorted similarity)
        if accepted_topic:
            correct_topics.append(score["topic"])
            crs.addstr(f"Accepted topic #{i}: ", curses.A_BOLD)
            crs.addstr(f"{score['topic']}\n")
            continue

        crs.addstr("\n")
        crs.addstr(f"Topic #{i}: ", curses.A_BOLD)
        crs.addstr(f"{score['topic']}\n")

        if user_says_accept(crs):
            correct_topics.append(score["topic"])
            accepted_topic = True
        else:
            pass  # rejection is implicit

    return correct_topics


def get_topics_to_check(data_sample, clean_data):
    topics_to_check = []
    warning = ''

    state = data_sample.get("state", NOT_VISITED)

    if state == NOT_VISITED:
        topics_to_check = data_sample["scores"]

    text_id = data_sample["text_id"]
    if state == CHECKED and text_id not in clean_data:
        topics_to_check = data_sample["scores"]
        warning = (f'WARNING: Sample {text_id} marked as CHECKED (2), but not present in CLEAN_DATA. Someone tampered with CLEAN_DATA?\n\n')

    return topics_to_check, warning


def redo_if_needed(sorted_topics, correct_topics, data_sample, text_displayer, crs):
    current_text = {
        "text": data_sample["text"],
        "topics": correct_topics,
        "potential_hard_negatives": data_sample["potential_hard_negatives"],
    }

    while True:
        addstr_wordwrap(
            crs,
            "\n\nIf you want to continue, press 'c', to redo an annotation press 'r', to quit press 'q'. ",
            0,
        )
        key = crs.getch()
        if key in (ord(c) for c in "qQcCrRsS"):
            crs.addstr("\n")
            if key == ord("q") or key == ord("Q"):  # Quit
                crs.addstr("Are you sure you want to quit? [Y/n] ")
                key = crs.getch()
                if key == ord("y") or key == ord("Y"):
                    raise QuitError()
            elif key == ord("r") or key == ord("R"):  # Redo
                addstr_wordwrap(
                    crs,
                    "Choose which annotation to redo by pressing the number of the annotation: ",
                    0,
                )
                annot_id_str = chr(crs.getch())
                crs.addstr("\n")
                if not annot_id_str.isnumeric() or int(annot_id_str) not in range(1, len(sorted_topics) + 1):
                    crs.addstr("Invalid annotation number.\n")
                    continue
                else:
                    annot_id = int(annot_id_str)
                    score = sorted_topics[annot_id - 1]

                    crs.addstr("Relevant? [Y/n] ")
                    choice = crs.getch()
                    while choice not in [ord("y"), ord("Y"), ord("n"), ord("N")]:
                        choice = crs.getch()

                    if choice == ord("n") or choice == ord("N"):
                        if score["topic"] in current_text["topics"]:
                            current_text["topics"].remove(score["topic"])
                    elif choice == ord("y") or choice == ord("Y"):
                        if score["topic"] not in current_text["topics"]:
                            current_text["topics"].append(score["topic"])
                    sorted_topics[annot_id - 1] = score
                    crs.clear()
                    crs.refresh()
                    text_displayer.display()
                    display_topics(sorted_topics, current_text, crs)

            elif key == ord("c") or key == ord("C"):
                data_sample["state"] = CHECKED
                break
            elif key == ord("s") or key == ord("S"):

                crs.addstr("\nAre you sure you want to skip this text? [Y/n] ")
                key = crs.getch()
                while key not in [ord("y"), ord("Y"), ord("n"), ord("N")]:
                    key = crs.getch()

                crs.addstr("\n")
                if key == ord("y") or key == ord("Y"):
                    data_sample["state"] = SKIPPED
                    raise SkipError

    return current_text


def main():
    args = get_args()

    with open(args.INPUT_FILE, "r") as f:
        lines = f.readlines()

    if os.path.exists(args.CLEAN_DATASET):
        with open(args.CLEAN_DATASET, "r") as f:
            clean_data = json.load(f)
    else:
        clean_data = {}

    nb_texts = len(lines)
    nb_texts_cleaned = len(clean_data)
    cleaned_texts_this_session = 0

    with CursesWindow() as crs:
        put_introduction(nb_texts, nb_texts_cleaned, crs)
        key = crs.getch()

        if key != ord("c"):
            print(f'DEBUG got {key}')
            return 0

        crs.clear()
        crs.refresh()

        for i, line in enumerate(lines):
            data_sample = json.loads(line)

            topics_to_check, warning = get_topics_to_check(data_sample, clean_data)
            if not topics_to_check:
                continue
            if warning:
                crs.addstr(warning)

            text_displayer = TextDisplayer(data_sample["text"], nb_texts - (len(clean_data)), cleaned_texts_this_session, crs)
            text_displayer.display()

            sorted_topics = sorted(topics_to_check, key=lambda x: x["similarity"], reverse=False)

            try:
                correct_topics = annotate_topics(sorted_topics, crs)
            except SkipError:
                data_sample["state"] = SKIPPED
                continue
            finally:
                crs.clear()
                crs.refresh()

            text_displayer.display()
            display_topics(
                sorted_topics,
                {"text": data_sample["text"], "topics": correct_topics},
                crs,
            )

            # Redo annotations if needed, quit or continue
            end = False
            try:
                current_text = redo_if_needed(sorted_topics, correct_topics, data_sample, text_displayer, crs)
            except QuitError:
                end = True
            except SkipError:
                pass  # already handled inside of redo_if_needed, to be improved

            # Update cleaned data
            clean_data[data_sample["text_id"]] = current_text
            cleaned_texts_this_session += 1

            # Clean data
            json.dump(
                clean_data,
                open(args.CLEAN_DATASET, "w"),
                indent=4,
                ensure_ascii=False,
            )

            # Original data, write back to file
            lines[i] = json.dumps(data_sample) + "\n"
            with open(args.INPUT_FILE, "w") as f:
                for line in lines:
                    f.write(line)

            if end:
                break

            crs.clear()
            crs.refresh()

        crs.getch()  # TODO DEBUG


if __name__ == '__main__':
    main()
