import json

from openai import OpenAI

from evaluate_topic_modelling import TopicEvaluator, BasicMetric, CrossEncoderMetric
from utils import TopicGenerationLogger, get_annotations


class GptGenerator:
    def __init__(self):
        self.client = OpenAI()
        self.temperature = 0.2
        self.max_tokens = 64
        self.top_p = 1
        self.frequency_penalty = 0
        self.presence_penalty = 0
        self.system_message = ("You are Czech lingual expert with years of experience. You are providing 3 to 5 topics "
                               "that occur in a given text. Each topic has to make sense on its own and cover "
                               "substantial part of the text. The length of each topic never exceeds 4 words limit. "
                               "Answer in Czech language. Answer should contain only new line separated "
                               "topics.\n\nHere are some examples.\n\nTextA: \n<b>Italie, jmenovitě severní, "
                               "má od dob Karla IV. s Čechami styky a v době, kdy archa chuděnická vznikla, "
                               "působil v Praze a v bratrstvu malířském Vlach Romanus z Florencie.</b>\nText topics: "
                               "\nmezistátní vztahy\nmalířske bratstvo\n\nTextB: \nChce-li poznačiti nástroj, "
                               "ukazuje na domnělý předmět jak se v ruce drží. Tak na př. ukazuje spůsob, jak se drží "
                               "nůžky, nůž, lžíce, péro, píla, sekera, nebozez a t. d. a jaké pohyby s těmito "
                               "nástroji se provádějí.\nText topics: \nnakládání s nástroji\npohyby s "
                               "nástroji\n\nTextC:\nNěmecká věda nepomohla Lvu Nikolajeviči, protože požadavky, "
                               "jež on na tuto vědu kladl, byly příliš vysoké.\nText topics:\ndecké bádání "
                               "Tolstého\nVýzkum Lva Nikolajeviče Tolstého\n\n")

    def __call__(self, example_text, *args, **kwargs):
        generation_result = self.client.chat.completions.create(
            model="gpt-4-0125-preview",
            messages=[
                {
                    "role": "system",
                    "content": self.system_message
                },
                {
                    "role": "user",
                    "content": example_text
                }
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty
        )
        generated_answer = generation_result.choices[0].message.content
        return generated_answer.split("\n")

    def get_settings_repr(self):
        str_repr = "GPT4 model with settings:"
        str_repr += ('\n'.join(f"{attr}: {getattr(self, attr)}" for attr in dir(self) if not attr.startswith('__')))
        return str_repr


if __name__ == "__main__":
    max_topic_generations = 10
    save_to_file = True

    # topic_generator should be callable which takes text as argument
    # and returns list of topics on call
    topic_generator = GptGenerator()
    evaluator = TopicEvaluator(BasicMetric(), CrossEncoderMetric())
    logger = TopicGenerationLogger(topic_generator, evaluator, max_topic_generations)

    logger.print_settings()
    # Generating topics using topic_generator
    generated_all = []
    for text, topics in get_annotations("out.json", num_iterations=max_topic_generations):
        generated = {
            "text": text,
            "annotator_topics": topics,
            "generated_topics": topic_generator(text)
        }
        generated_all.append(generated)
        logger.new_generated(generated)

    logger.finished_generation(generated_all)

    # Evaluation
    result = evaluator.get_results(generated_all)
    logger.print_results(result)

