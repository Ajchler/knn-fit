from openai import OpenAI
import json, re, argparse
from pathlib import Path

R_EXTRACT = re.compile(r"```json\n(.*)\n```", re.DOTALL)

model = "gpt-4-turbo"

system = "Jsi expert na lingvistiku, historii, kulturu, právo, technologii a teologii."

# v češtině dělá méně gramatických chyb
prompt_A = """Zde je vstupní text. Vytvoř 5 MÍRNĚ chybných popisů textu.
Každý popis bude mít nejvýše 3 slova. Nevytvářej zjevně chybné popisy.
Všechny popisy musí být gramaticky správné a musí být známými frázemi, koncepty nebo odbornými termíny. Nevymýšlej nová slovní spojení.
Zobecnění správného popisu nebo nepřímý popis obsahu se nepočítají jako chybný popis - nevypisuj je.
Výsledné popisy by měly být schopny oklamat odborníka v oboru.
Výstupem bude správně formátovaný JSON objekt se dvěma klíči: 'descriptions' a 'explanations'.
'descriptions' je seznam popisů.
'explanations' je seznam vysvětlení proč každý popis splňuje tato kritéria v maximálně 8 slovech.
Nevypisuj formátování Prism."""

def spam_api(texts:list[tuple[str, str]], take:int=10):
    try:
        client = OpenAI()
    except:
        print("OpenAI API key not found. Please set it as an environment variable.")
        exit(-1)

    print("{")
    for i, (id, text) in enumerate(texts):
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": f"""{prompt_A}
                Input text: {text}"""}
            ]
        )

        if i != 0:
            print(",")
        print(f"\"{id}\": {completion.choices[0].message.content}")

        i += 1
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