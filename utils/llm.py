from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, AutoTokenizer, AutoModel, DynamicCache
import torch
from collections import Counter
import numpy as np
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer


def temperature_scaling(logits, temperature=1):
    logits = np.array(logits)
    logits /= temperature
    # apply softmax
    try:
        logits -= logits.max()
    except:
        logits = logits
    logits = logits - np.log(np.sum(np.exp(logits)))
    smx = np.exp(logits)
    return smx


class LLM:
    def __init__(self, model_name, generation_config=None):
        self.device = torch.device("mps" if torch.mps.is_available() else "cpu")
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name, model_max_length=512)
        self.generation_config = generation_config or {}

        self.model.to(self.device)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, prompt, return_logits=False):
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(self.device)
        input_ids = inputs['input_ids'].to(self.device)

        gen_kwargs = {
            "max_length": 512,
            "min_length": 10,
            "num_return_sequences": 1,
            "do_sample": True,
            "temperature": 0.9,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        generated_ids = self.model.generate(input_ids, **gen_kwargs).to(self.device)

        generated_sequences = []
        for sequence in generated_ids:
            text = self.tokenizer.decode(
                sequence,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            generated_sequences.append(text)

        if return_logits:
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, decoder_input_ids=generated_ids[:, :-1])
                logits = outputs.logits
                special_tokens = self.tokenizer.all_special_ids
                mask = torch.ones_like(generated_ids[0, 1:], dtype=torch.bool)
                for token_id in special_tokens:
                    mask &= (generated_ids[0, 1:] != token_id)
                filtered_logits = logits[:, mask]
            return generated_sequences, filtered_logits
        else:
            return generated_sequences

    def filter_logits(self, logits, words, use_softmax=True):
        token_ids = []
        for word in words:
            word_tokens = self.tokenizer.tokenize(word)
            word_token_ids = self.tokenizer.convert_tokens_to_ids(word_tokens)
            token_ids.extend(word_token_ids)

        token_ids = [t.item() for t in torch.tensor(token_ids, device=self.device)]
        count_tokens = dict(Counter(token_ids).most_common())
        token_ids_target = [key for key in count_tokens.keys() if count_tokens[key] == 1]
        filtered_logits = [logits[t].item() for t in token_ids_target]
        if use_softmax:
            filtered_logits = temperature_scaling(filtered_logits)
        return dict(zip(words, filtered_logits))



if __name__ == "__main__":
    print(torch.mps.is_available())
    model = LLM("t5-base", {"max_length": 50, "num_return_sequences": 1})
    print(model.generate("Choose one letter A/B/C?"))
    answer, logits = model.generate("Choose one letter A/B/C?", return_logits=True)
    filtered_logits = model.filter_logits(logits[0][1], words=["A", "B", "C", "D", 'a', 'b', 'c', 'd', '1', '2', '3', '4'])
    print(filtered_logits)
