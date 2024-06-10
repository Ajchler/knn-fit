from openai import OpenAI
import json, re, argparse
from pathlib import Path
import sys


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
    Nevypisuj formátování Prism."""
    }

    prompt_A = prompts["alternative"]

    def __init__(self, path):
        self.data_path = path
        self.data = json.load(open(path, mode="r"))
        already_generated = 0
        for i, (id, text) in enumerate(self.data.items()):
            if "llm_generated_hn" in text:
                already_generated += 1

        print(f"There are {already_generated} already generated hard negatives"
              f" and {len(self.data) - already_generated} to generate.")

        self.client = OpenAI()

    def spam_api(self, take: int = 10):
        print(f"Generating hard negatives for {take} texts.")
        generated = 0
        for id, text in self.data.items():
            if "llm_generated_hn" in text:
                continue

            if generated == take:
                break

            prompt = f"{OpenAIGeneration.prompt_A}\nVstupní text: {text['text']}"
            completion = self.client.chat.completions.create(
                model=OpenAIGeneration.model,
                messages=[
                    {
                        "role": "system",
                        "content": OpenAIGeneration.system
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            result = json.loads(completion.choices[0].message.content)
            print(f"Raw response: {result}")
            if OpenAIGeneration.prompt_A == "alternative":
                result = [item for sublist in result.values() for item in sublist]
            self.data[id]['llm_generated_hn'] = result
            print(f"Generated hard negatives for text {id}.")
            generated += 1

        if generated < take:
            print(f"Hard negatives for only {generated} texts generated.")

        json.dump(self.data, open(self.data_path, mode="w"), indent=4, ensure_ascii=False)


if "__main__" == __name__:
    # argparse to merge two json files
    # argparse option to specify how many hard negatives to take in sum
    # argparse to start annotating hard negatives
    parser = argparse.ArgumentParser(description="Tool for texts hard negatives generation, "
                                                 "finding in dataset and annotating.")
    # argparse to create hard negatives - options [generate, find, both]
    parser.add_argument("--generate-hn", type=str, choices=["generate", "find", "both"],
                        default=None, help="Generate hard negatives for texts.")
    parser.add_argument("--source", type=Path, help="Source file to clean dataset. "
                                                    "Default is data/clean_dataset.json."
                                                    "Has to be dictionary with keys as text ids and values as text "
                                                    "dictionaries.",
                        default="data/clean_dataset.json", nargs="?")
    parser.add_argument("--take", dest="take", type=int, default=10,
                        help="Number of texts to generate hard negatives for.")



    args = parser.parse_args()

    if args.generate_hn is not None:
        print(f"Generating hard negatives for {args.source}.")
        if args.generate_hn == "generate" or args.generate_hn == "both":
            src_path = Path(args.source)
            if not src_path.exists():
                print(f"File {args.source} not found.")
                exit(-1)

            print("Calling OpenAI API to generate hard negatives.")
            OpenAIGeneration(src_path).spam_api(args.take)

        if args.generate_hn == "find" or args.generate_hn == "both":
            pass
