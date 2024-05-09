from openai import OpenAI
import json, re, argparse
from pathlib import Path
import sys

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

def spam_api(texts:list[tuple[str, str]], take:int=10):
    try:
        client = OpenAI()
    except:
        print("OpenAI API key not found. Please set it as an environment variable.", file=sys.stderr)
        exit(-1)

    print("{")
    for i, (id, text) in enumerate(texts):
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": f"""{prompt_A}
                Vstupní text: {text}"""}
            ]
        )

        if i != 0:
            print(",")
        print(f"\"{id}\": {completion.choices[0].message.content}")

        if i == take:
            break
    print("}")

def extract_json(source:Path)->list[tuple[str,str]]:
    #[x for x in [y for y in v.values()][1]["topics"] if x]
    with open(source) as f:
        topics = json.load(f)
        ret = [(k, v["text"]) for k, v in topics.items()]
    return ret

def main():
    argp = argparse.ArgumentParser(description="Generate hard negatives for texts.")
    argp.add_argument("source", type=Path, help="Source file with texts.")
    argp.add_argument("--take", dest="take", type=int, default=10, help="Number of texts to generate hard negatives for.")
    args = argp.parse_args()

    src_path = Path(args.source)
    if not src_path.exists():
        print(f"File {args.source} not found.")
        exit(-1)

    data = extract_json(src_path)
    spam_api(data, args.take)

if __name__ == "__main__":
    main()