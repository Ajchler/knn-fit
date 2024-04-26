import json
import curses
import os

if __name__ == '__main__':
    with open('out.json', 'r') as f:
        data = json.load(f)

    crs = curses.initscr()

    annotated_data = {}
    counter = 0

    while True:
        crs.addstr(f"Currently there are {len(annotated_data)} texts annotated.\n")
        crs.addstr('If you want to go through the texts again, press "c", to quit pres "q".\n')
        key = crs.getch()
        if key == 113:
            break
        elif key == 99:
            crs.clear()
            crs.refresh()

        text_counter = 1
        total_texts = len(data)

        for d in data:
            crs.addstr(f"Text {text_counter}/{total_texts}\n")
            text_counter += 1
            text_topics = []
            reviewed_topics = []
            for key in data[d]:
                if key == 'text':
                    text = data[d][key]
                else:
                    for t in data[d][key]['topics']:
                        if t not in text_topics and t != '':
                            text_topics.append(t)

            if len(text_topics) == 0:
                continue

            if text in annotated_data:
                continue

            crs.addstr(f"Currently there are {len(annotated_data)} texts annotated.\n")

            crs.addstr('Text:\n')
            crs.addstr(text + '\n\n')
            crs.addstr('Topics:\n')
            for t in text_topics:
                crs.addstr(t + '\n')
            crs.addstr('\n')

            crs.addstr('If you want to review the next text, press "n", to review press "c" or "q" to quit.\n')
            crs.refresh()

            key = crs.getch()
            if key == 110:
                crs.clear()
                continue
            elif key == 113:
                break

            crs.addstr('\nFor each topic, press "0" if the topic is relevant to the text, or "1" if it is not.\n\n')
            for t in text_topics:
                crs.addstr('\n' + t + '\n')
                crs.refresh()
                while True:
                    key = crs.getch()
                    if key == 48:
                        reviewed_topics.append((t, 0))
                        break
                    elif key == 49:
                        reviewed_topics.append((t, 1))
                        break
                    else:
                        continue

            crs.clear()
            crs.refresh()
            topics_dict = {t: r for t, r in reviewed_topics}
            annotated_data[d] = {
                'text': text,
                'topics': topics_dict
            }

    crs.clear()
    curses.endwin()
    json.dump(annotated_data, open('annotated_dataset.json', 'w'), indent=4)
