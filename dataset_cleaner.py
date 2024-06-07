import curses
import json
import os

with open("evaluation-data/out-mlm-mpnet-base-v2.json", "r") as f:
    data = json.load(f)

if os.path.exists("data/clean_dataset.json"):
    with open("data/clean_dataset.json", "r") as f:
        clean_data = json.load(f)
else:
    clean_data = {}

crs = curses.initscr()

crs.addstr("***********************************\n")
crs.addstr("* Welcome to the dataset cleaner! *\n")
crs.addstr("***********************************\n\n\n")

crs.addstr("Instructions:\n\n", curses.A_BOLD)
crs.addstr("You will be presented with texts and potential topics for each text.\n")
crs.addstr(
    "The topics are sorted by similarity to the text with the least similar topic first.\n"
)
crs.addstr("For each topic you will be prompted to mark it as relevant or not.\n")
crs.addstr("Press y/Y if the topic is relevant, n/N if it is not.\n")
crs.addstr(
    "Once you mark a topic relevant, the rest of the topics will be marked as relevant as well.\n\n\n"
)
crs.addstr("If you want to start cleaning, press 'n' or 'q' to quit.\n\n")
key = crs.getch()

if key == ord("n"):
    crs.clear()
    crs.refresh()

    for text_id in data:
        accepted_topic = False

        correct_topics = []

        # Check if text_id is already in clean_data
        if text_id in clean_data.keys():
            continue

        crs.addstr("\nText:\n\n", curses.A_BOLD)
        text = data[text_id]["text"]
        scores = data[text_id]["scores"]
        sorted_scores = sorted(scores, key=lambda x: x["similarity"], reverse=True)

        crs.addstr(text + "\n")

        for score in sorted_scores:
            if accepted_topic:
                correct_topics.append(score["topic"])
                crs.addstr(f"Accepted topic: {score['topic']}\n")
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

        while True:
            crs.addstr("If you want to continue, press 'c', to quit press 'q'.\n")
            key = crs.getch()
            if key == ord("q") or key == ord("Q") or key == ord("c") or key == ord("C"):
                break

        crs.clear()
        crs.refresh()


crs.clear()
curses.endwin()
