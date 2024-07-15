#!/usr/bin/env python3

import argparse
import curses
import json
import os

from utils import (
    addstr_wordwrap,
    CursesWindow,
    curses_overflow_restarts,
    ScreenOwner,
    print_job_done,
)
import getting_user_input

NOT_VISITED = 0
SKIPPED = 1
CHECKED = 2


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--INPUT_FILE",
        required=False,
        default="evaluation-data/out-mlm-mpnet-base-v2-all-texts_example.jsonl",
    )
    parser.add_argument("--CLEAN_DATASET", default="data/clean_dataset.json")

    return parser.parse_args()


class ScreenOwnerCleaning(ScreenOwner):
    controls_string = (
        "Press y/Y if the topic is relevant, n/N if it is not. You can also skip this text anytime by "
        "pressing 's'.\nYou will also be able to redo the current text after last topic if you make a "
        "mistake during cleaning.\n\n"
    )

    def __init__(self, crs, text, nb_left, nb_cleaned_this_session, sorted_topics):

        self.correct_topics = None
        self.sorted_topics = sorted_topics

        super().__init__(crs, text, nb_left, nb_cleaned_this_session)

    def redraw(self):
        super().redraw()

        if self.correct_topics is not None:
            for i, topic in enumerate(self.sorted_topics, start=1):
                self.crs.addstr(f"Topic #{i}: ", curses.A_BOLD)
                self.crs.addstr(f"{topic['topic']}\n")
                self.crs.addstr("Relevant? ")
                if topic["topic"] in self.correct_topics:
                    self.crs.addstr("✓\n\n")
                else:
                    self.crs.addstr("✗\n\n")

    def update_correct_topics(self, correct_topics):
        self.correct_topics = correct_topics
        self.redraw()


def put_introduction(nb_texts, nb_cleaned, crs):
    crs.addstr("***********************************\n")
    crs.addstr("* Welcome to the dataset cleaner! *\n")
    crs.addstr("***********************************\n\n\n")

    crs.addstr("Statistics:\n", curses.A_BOLD)
    crs.addstr(f"Number of texts: {nb_texts}\n")
    crs.addstr(f"Number of cleaned texts: {nb_cleaned}\n")
    crs.addstr(f"Number of texts left: {nb_texts - nb_cleaned}\n\n\n")

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


def annotate_topics(sorted_topics, crs):
    accepted_topic = False
    correct_topics = []

    for i, score in enumerate(sorted_topics, start=1):
        #  Once a topic is accepted, we accept all the following ones (based on sorted similarity)
        if accepted_topic:
            correct_topics.append(score["topic"])
            continue

        crs.addstr("\n")
        crs.addstr(f"Topic #{i}: ", curses.A_BOLD)
        crs.addstr(f"{score['topic']}\n")

        if getting_user_input.accept_or_reject(crs, "Relevant? [Y/n] "):
            correct_topics.append(score["topic"])
            accepted_topic = True
        else:
            pass  # rejection is implicit

    return correct_topics


def get_topics_to_check(data_sample, clean_data):
    topics_to_check = []
    warning = ""

    state = data_sample.get("state", NOT_VISITED)

    if state == NOT_VISITED:
        topics_to_check = data_sample["scores"]

    text_id = data_sample["text_id"]
    if state == CHECKED and text_id not in clean_data:
        topics_to_check = data_sample["scores"]
        warning = f"WARNING: Sample {text_id} marked as CHECKED (2), but not present in CLEAN_DATA. Someone tampered with CLEAN_DATA?\n\n"

    return topics_to_check, warning


def annotation_to_redo(nb_topics, crs):
    addstr_wordwrap(
        crs,
        "Choose which annotation to redo by pressing the number of the annotation: ",
        0,
    )
    annot_id_str = chr(crs.getch())
    crs.addstr("\n")
    if not annot_id_str.isnumeric() or int(annot_id_str) not in range(1, nb_topics + 1):
        return None

    return int(annot_id_str) - 1


def redo_if_needed(sorted_topics, correct_topics, screen_owner, crs):
    while True:
        addstr_wordwrap(
            crs,
            "\n\nPress 'c' to continue, 'r' to redo if you made a mistake, 'q' to quit. ",
            0,
        )
        action = getting_user_input.redo_or_proceed(crs)
        if action == "redo":  # Redo
            annot_id = annotation_to_redo(len(sorted_topics), crs)
            if annot_id is None:
                crs.addstr("Invalid annotation number.\n")
                continue

            topic = sorted_topics[annot_id]

            relevant = getting_user_input.redo_accept(crs)

            if relevant:
                # User marked as relevant topic which was already marked as relevant
                if topic["topic"] not in correct_topics:
                    correct_topics.append(topic["topic"])
            elif topic["topic"] in correct_topics:
                correct_topics.remove(topic["topic"])

            sorted_topics[annot_id] = topic
            screen_owner.update_correct_topics(correct_topics)

        elif action == "continue":
            break

    return correct_topics


def create_rejected_topics(correct_topics, sorted_topics):
    return [
        {"topic": topic["topic"], "type": "rejected"}
        for topic in sorted_topics
        if topic["topic"] not in correct_topics
    ]


@curses_overflow_restarts
def start_data_cleaning(clean_data, lines, args):
    nb_texts = len(lines)
    nb_texts_cleaned = len(clean_data)
    remaining = 0
    for line in lines:
        data_sample = json.loads(line)
        state = data_sample.get("state", NOT_VISITED)
        if state == NOT_VISITED:
            remaining += 1
    cleaned_texts_this_session = 0

    with CursesWindow() as crs:
        put_introduction(nb_texts, nb_texts_cleaned, crs)

        quit_or_proceed = getting_user_input.quit_or_proceed(crs)
        if quit_or_proceed == "quit":
            return 0

        for i, line in enumerate(lines):
            data_sample = json.loads(line)

            topics_to_check, warning = get_topics_to_check(data_sample, clean_data)
            if not topics_to_check:
                continue
            if warning:
                crs.addstr(warning)

            sorted_topics = sorted(
                topics_to_check, key=lambda x: x["similarity"], reverse=False
            )
            screen_owner = ScreenOwnerCleaning(
                crs,
                data_sample["text"],
                remaining,
                cleaned_texts_this_session,
                sorted_topics,
            )

            skipped = False
            correct_topics = []
            try:
                correct_topics = annotate_topics(sorted_topics, crs)
                screen_owner.update_correct_topics(correct_topics)
            except getting_user_input.SkipError:
                skipped = True

            # Redo annotations if needed, quit or continue
            end = False
            try:
                if not skipped:
                    correct_topics = redo_if_needed(
                        sorted_topics, correct_topics, screen_owner, crs
                    )
            except getting_user_input.QuitError:
                end = True
            except getting_user_input.SkipError:
                skipped = True

            # Original data, write back to file with updated state
            data_sample["state"] = CHECKED if not skipped else SKIPPED
            lines[i] = json.dumps(data_sample, ensure_ascii=False) + "\n"
            with open(args.INPUT_FILE, "w") as f:
                for line in lines:
                    f.write(line)

            # Do not update cleaned data if the text was skipped
            if skipped:
                continue

            # Add user rejected topics to the set of potential hard negatives
            rejected_topics = create_rejected_topics(correct_topics, sorted_topics)
            new_potential_hns = (
                data_sample["potential_hard_negatives"] + rejected_topics
            )

            new_annotated = {
                "text": data_sample["text"],
                "topics": correct_topics,
                "potential_hard_negatives": new_potential_hns,
            }

            # Update cleaned data
            clean_data[data_sample["text_id"]] = new_annotated
            cleaned_texts_this_session += 1
            remaining -= 1

            # Clean data
            json.dump(
                clean_data,
                open(args.CLEAN_DATASET, "w"),
                indent=4,
                ensure_ascii=False,
            )

            if end:
                break

        if remaining == 0:
            print_job_done(crs)


def main():
    args = get_args()

    with open(args.INPUT_FILE, "r") as f:
        lines = f.readlines()

    if os.path.exists(args.CLEAN_DATASET):
        with open(args.CLEAN_DATASET, "r") as f:
            clean_data = json.load(f)
    else:
        clean_data = {}

    start_data_cleaning(clean_data, lines, args)


if __name__ == "__main__":
    main()
