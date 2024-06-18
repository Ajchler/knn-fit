import curses
import json
import os

from utils import addstr_wordwrap

NOT_VISITED = 0
SKIPPED = 1
CHECKED = 2

INPUT_FILE = "evaluation-data/out-mlm-mpnet-base-v2-all-texts.json"

with open(INPUT_FILE, "r") as f:
    data = json.load(f)

if os.path.exists("data/clean_dataset.json"):
    with open("data/clean_dataset.json", "r") as f:
        clean_data = json.load(f)
else:
    clean_data = {}

try:
    crs = curses.initscr()
except:
    print("Error initializing curses, try increasing the terminal size.")
    exit(1)

number_of_texts = len(data)
number_of_cleaned_texts = len(clean_data)

crs.addstr("***********************************\n")
crs.addstr("* Welcome to the dataset cleaner! *\n")
crs.addstr("***********************************\n\n\n")

crs.addstr("Statistics:\n", curses.A_BOLD)
crs.addstr(f"Number of texts: {number_of_texts}\n")
crs.addstr(f"Number of cleaned texts: {number_of_cleaned_texts}\n")
crs.addstr(f"Number of texts left: {number_of_texts - number_of_cleaned_texts}\n\n\n")

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
key = crs.getch()

cleaned_texts_this_session = 0

if key == ord("c"):
    end = False
    crs.clear()
    crs.refresh()

    for text_id in data:
        accepted_topic = False

        correct_topics = []

        # Check whether there are topics to be annotated
        annotate = False
        scores_to_check = []
        for score in data[text_id]["scores"]:
            try:
                state = score["state"]
            except KeyError:
                state = NOT_VISITED
            if state == NOT_VISITED or state == SKIPPED:
                annotate = True
                scores_to_check.append(score)

        if not annotate:
            continue

        crs.addstr("Statistics:\n", curses.A_BOLD)
        crs.addstr(
            f"You have cleaned {cleaned_texts_this_session} texts this session.\n"
        )
        crs.addstr(f"There are {number_of_texts - (len(clean_data))} texts left.\n\n")

        crs.addstr("Controls:\n", curses.A_BOLD)
        controls_string = "Press y/Y if the topic is relevant, n/N if it is not. You can also skip any topic by pressing 's' if you are unable to make a decision.\nYou will also be able to redo the current text after last topic if you make a mistake during cleaning.\n\n"
        addstr_wordwrap(crs, controls_string, 0)
        crs.addstr("\nText:\n\n", curses.A_BOLD)
        text = data[text_id]["text"]
        addstr_wordwrap(crs, text + "\n", 0)
        crs.addstr("\n")
        scores = scores_to_check
        sorted_scores = sorted(scores, key=lambda x: x["similarity"], reverse=False)

        count = 0
        flagged_scores = []

        # Annotate topics
        for score in sorted_scores:
            count += 1
            if accepted_topic:
                correct_topics.append(score["topic"])
                crs.addstr(f"Accepted topic #{count}: ", curses.A_BOLD)
                crs.addstr(f"{score['topic']}\n")
                score["state"] = CHECKED
                flagged_scores.append(score)
                continue
            crs.addstr("\n")
            crs.addstr(f"Topic #{count}: ", curses.A_BOLD)
            crs.addstr(f"{score['topic']}\n")
            crs.addstr("Relevant? [Y/n/s] ")
            key = crs.getch()
            while key not in [
                ord("y"),
                ord("Y"),
                ord("n"),
                ord("N"),
                ord("s"),
                ord("S"),
            ]:
                key = crs.getch()
            if key == ord("n") or key == ord("N"):
                score["state"] = CHECKED
            elif key == ord("y") or key == ord("Y"):
                correct_topics.append(score["topic"])
                accepted_topic = True
                score["state"] = CHECKED
            elif key == ord("s") or key == ord("S"):
                score["state"] = SKIPPED
            flagged_scores.append(score)
            crs.addstr("\n")

        current_text = {}
        current_text["text"] = text
        current_text["topics"] = correct_topics

        # Redo annotations if needed, quit or continue
        while True:
            addstr_wordwrap(
                crs,
                "\n\nIf you want to continue, press 'c', to redo an annotation press 'r', to quit press 'q'. ",
                0,
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
                        break
                elif key == ord("r") or key == ord("R"):
                    addstr_wordwrap(
                        crs,
                        "\nChoose which annotation to redo by pressing the number of the annotation: ",
                        0,
                    )
                    key = chr(crs.getch())
                    if not key.isnumeric() or int(key) not in range(1, count + 1):
                        crs.addstr("\nInvalid annotation number.")
                        continue
                    else:
                        state = "not relevant\n"
                        key = int(key)
                        score = sorted_scores[key - 1]
                        if score["topic"] in current_text["topics"]:
                            current_text["topics"].remove(score["topic"])
                        else:
                            current_text["topics"].append(score["topic"])
                            state = "relevant\n"
                        state_str = (
                            f"\nAnnotation {key} changed and is now marked as " + state
                        )
                        addstr_wordwrap(crs, state_str, 0)

                elif key == ord("c") or key == ord("C"):

                    break

        # Update cleaned data
        clean_data[text_id] = current_text
        cleaned_texts_this_session += 1

        # Update original data with flags
        data[text_id]["scores"] = flagged_scores

        # Clean data
        json.dump(
            clean_data,
            open("data/clean_dataset.json", "w"),
            indent=4,
            ensure_ascii=False,
        )

        # Original data
        json.dump(
            data,
            open(INPUT_FILE, "w"),
            indent=4,
            ensure_ascii=False,
        )

        if end:
            break

        crs.clear()
        crs.refresh()


crs.clear()
curses.endwin()
