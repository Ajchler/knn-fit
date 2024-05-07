import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def clean_dataset():
    f = open('data/out.json')

    data = json.load(f)
    data_list = []
    for text_id, annotated_text in data.items():
        annotation_ids = [k for k in annotated_text.keys() if k not in ["text", "text_id"]]
        annotations = []
        for an_id in annotation_ids:
            annotations.append(annotated_text[an_id])
        for annotation in annotations:
            new_text = {
                'text_id': text_id,
                'text': annotated_text['text'],
                'user_id': annotation["user_id"],
                'user_topics': [t for t in annotation["topics"] if t != ""],
            }
            if len(new_text['user_topics']) != 0:
                data_list.append(new_text)

    with open("data/out-clean.json", "w") as f:
        json.dump(data_list, f, ensure_ascii=False, indent=4)

    # Create DataFrame
    df = pd.DataFrame(data_list)
    df


def create_text_embeddings(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model = model.to(device)

    batch_size = 256

    outputs = []
    texts_nr = len(df_texts["text"])
    num_batches = texts_nr // batch_size
    for i in range(num_batches + 1):
        start_idx = i * batch_size
        end_idx = np.minimum((i + 1) * batch_size, len(df_texts["text"]))
        print(f"Processing from {i * batch_size}-{end_idx}/{texts_nr}.")
        batch = df_texts["text"].tolist()[start_idx:end_idx]
        inputs_d = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
        inputs_d.to(device)

        with torch.no_grad():
            outputs_d = model(**inputs_d)

        outputs.append(outputs_d.pooler_output)

        del inputs_d, outputs_d

        # Clear GPU cache
        torch.cuda.empty_cache()

    text_embeddings = torch.cat(outputs, 0)
    print(text_embeddings)
    print(text_embeddings.shape)

    torch.save(text_embeddings, f"text_embeddings_{model_name_file}.pt")
    return text_embeddings


def flatten(xss):
    return [x for xs in xss for x in xs]


if __name__ == "__main__":
    df_texts = pd.read_json("out-clean.json")
    model_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    model_name_file = model_name.replace("/", "_")

    regenerate_embeddings = False
    if regenerate_embeddings:
        text_embeddings = create_text_embeddings(model_name)
    else:
        text_embeddings = torch.load(f"text_embeddings_{model_name_file}.pt")

    text_embeddings = text_embeddings / text_embeddings.norm(dim=1)[:, None]
    similarity = text_embeddings @ text_embeddings.transpose(0, 1)
    similarity = similarity.cpu().numpy()
    k = 4

    similar_texts = []
    for i in range(len(similarity)):
        # -1 because self should always be on last position
        top_k = np.argpartition(similarity[i], -k)[-k:]
        top_k_sorted = top_k[np.argsort(-similarity[i][top_k])]

        most_similar = []
        for most_similar_idx in top_k_sorted[1:]:
            most_similar.append(
                {
                    "text": df_texts.iloc[most_similar_idx]["text"],
                    "user_topics": df_texts.iloc[most_similar_idx]["user_topics"],
                    "cosine_sim": f"{similarity[i][most_similar_idx]:0.6f}"
                }
            )

        this_topics_set = set(df_texts.iloc[i]["user_topics"])
        best_text_topics_set = set(most_similar[0]["user_topics"])
        best_text_topics_set_all = set(flatten(list(map(lambda x: x["user_topics"], most_similar)))[:10])
        similar_texts.append(
            {
                "text": df_texts.iloc[i]["text"],
                "user_topics": df_texts.iloc[i]["user_topics"],
                "potential_negatives_one": list(best_text_topics_set - this_topics_set),
                "potential_negatives_all": list(best_text_topics_set_all - this_topics_set),
                "most_similar_texts": most_similar
            }
        )

    with open(f"neg_exSets_{model_name_file}.json", "w") as f:
        json.dump(similar_texts, f, indent=4, ensure_ascii=False)
        print(json.dumps(similar_texts[:5], indent=4, ensure_ascii=False))
