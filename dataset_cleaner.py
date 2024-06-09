import curses
import json
import os

with open("evaluation-data/out-mlm-mpnet-base-v2-all-texts.json", "r") as f:
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
crs.addstr("If you want to start cleaning, press 'c' or 'q' to quit.\n\n")
key = crs.getch()

cleaned_texts_this_session = 0

if key == ord("c"):
    end = False
    crs.clear()
    crs.refresh()

    for text_id in data:
        accepted_topic = False

        correct_topics = []

        # Check if text_id is already in clean_data
        if text_id in clean_data.keys():
            continue

        crs.addstr("Statistics:\n", curses.A_BOLD)
        crs.addstr(
            f"You have cleaned {cleaned_texts_this_session} texts this session.\n"
        )
        crs.addstr(f"There are {number_of_texts - (len(clean_data))} texts left.\n\n")

        crs.addstr("Controls:\n", curses.A_BOLD)
        crs.addstr("Press y/Y if the topic is relevant, n/N if it is not.\n\n")

        crs.addstr("\nText:\n\n", curses.A_BOLD)
        text = data[text_id]["text"]
        scores = data[text_id]["scores"]
        sorted_scores = sorted(scores, key=lambda x: x["similarity"], reverse=False)

        crs.addstr(text + "\n")

        for score in sorted_scores:
            if accepted_topic:
                correct_topics.append(score["topic"])
                crs.addstr(f"\nAccepted topic: {score['topic']}")
                continue
            crs.addstr("\n\n")
            crs.addstr("Topic: ", curses.A_BOLD)
            crs.addstr(f"{score['topic']}\n")
            crs.addstr("Relevant? [Y/n] ")
            key = crs.getch()
            if key == ord("n") or key == ord("N"):
                continue
            else:
                correct_topics.append(score["topic"])
                accepted_topic = True

        clean_data[text_id] = {}
        clean_data[text_id]["text"] = text
        clean_data[text_id]["topics"] = correct_topics

        cleaned_texts_this_session += 1

        json.dump(
            clean_data,
            open("data/clean_dataset.json", "w"),
            indent=4,
            ensure_ascii=False,
        )

        while True:
            crs.addstr("\n\nIf you want to continue, press 'c', to quit press 'q'. ")
            key = crs.getch()
            if key == ord("q") or key == ord("Q") or key == ord("c") or key == ord("C"):
                if key == ord("q") or key == ord("Q"):
                    crs.addstr("\nAre you sure you want to quit? [Y/n] ")
                    key = crs.getch()
                    if key == ord("y") or key == ord("Y"):
                        end = True
                        break
                else:
                    break

        if end:
            break

        crs.clear()
        crs.refresh()


crs.clear()
curses.endwin()
