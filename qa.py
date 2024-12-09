import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoTokenizer, AutoModelForSeq2SeqLM
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW
import pandas as pd
import re
from sklearn.model_selection import train_test_split

from utils.llm import LLM


class TaskQuestionAnswerDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]

        # For question generation
        task_input = f"generate question: {item['task']} context: {item['environment_full']}"
        question = f"question: {item['question']}"

        # For answer variants generation
        answer_input = f"generate answer variants: {item['question']}"
        answers = f"answers: {item['all_variants']}"

        # Tokenize inputs and targets
        encodings = {
            'task': self.tokenizer(
                task_input,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors="pt"
            ),
            'question': self.tokenizer(
                question,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors="pt"
            ),
            'answer_input': self.tokenizer(
                answer_input,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors="pt"
            ),
            'answers': self.tokenizer(
                answers,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors="pt"
            )
        }

        return {
            'task_input_ids': encodings['task']['input_ids'].squeeze(),
            'task_attention_mask': encodings['task']['attention_mask'].squeeze(),
            'question_labels': encodings['question']['input_ids'].squeeze(),
            'answer_input_ids': encodings['answer_input']['input_ids'].squeeze(),
            'answer_attention_mask': encodings['answer_input']['attention_mask'].squeeze(),
            'answer_labels': encodings['answers']['input_ids'].squeeze(),
        }


    def train_data(orig_data, rows = 100):
        dataset = pd.read_csv(orig_data, nrows=rows)
        # Calibration set
        data_train_200 = pd.DataFrame(
            columns=['environment_full', "task", "is_amb", "question", 'all_variants', 'answer']).astype({
            'environment_full': 'str',
            "task": 'str',
            "is_amb": 'bool',
            "question": 'str',
            'all_variants': 'str',
            'answer': 'str'
        })

        new_rows = []
        for i in range(len(dataset)):  # len(dataset)
            description = dataset.loc[i, 'environment_full']
            amb_task = dataset.loc[i, 'ambiguous_task']
            unamb_task = dataset.loc[i, 'unambiguous_indirect']
            question = dataset.loc[i, 'question']
            options = dataset.loc[i, 'variants']
            answer = dataset.loc[i, 'answer']

            new_rows.append({'environment_full': description, 'task': amb_task, 'is_amb': True, 'question': question,
                             'all_variants': options, 'answer': answer})
            new_rows.append({'environment_full': description, 'task': unamb_task, 'is_amb': False, 'question': question,
                             'all_variants': options, 'answer': answer})

        data_train_200 = pd.concat([data_train_200, pd.DataFrame(new_rows)], ignore_index=True)
        data_train_200.to_csv('data/data_train_200.csv')
        return 'data/data_train_200.csv'


# Training function
def train_model(model, train_dataloader, val_dataloader, num_epochs=3):
    optimizer = AdamW(model.parameters(), lr=3e-5)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in train_dataloader:
            # Question generation
            question_outputs = model(
                input_ids=batch['task_input_ids'].to(model.device),
                attention_mask=batch['task_attention_mask'].to(model.device),
                labels=batch['question_labels'].to(model.device)
            )

            # Answer variants generation
            answer_outputs = model(
                input_ids=batch['answer_input_ids'].to(model.device),
                attention_mask=batch['answer_attention_mask'].to(model.device),
                labels=batch['answer_labels'].to(model.device)
            )

            loss = question_outputs.loss + answer_outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                question_outputs = model(
                    input_ids=batch['task_input_ids'].to(model.device),
                    attention_mask=batch['task_attention_mask'].to(model.device),
                    labels=batch['question_labels'].to(model.device)
                )

                answer_outputs = model(
                    input_ids=batch['answer_input_ids'].to(model.device),
                    attention_mask=batch['answer_attention_mask'].to(model.device),
                    labels=batch['answer_labels'].to(model.device)
                )

                val_loss += (question_outputs.loss + answer_outputs.loss).item()

            print(f"Epoch {epoch + 1}")
            print(f"Average training loss: {total_loss / len(train_dataloader)}")
            print(f"Average validation loss: {val_loss / len(val_dataloader)}")


def main():
    name = TaskQuestionAnswerDataset.train_data('data/AmbiK_data.csv', rows=50)
    df = pd.read_csv("data/data_train_200.csv")
    print(df.columns)
    # Split dataset
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    # Initialize tokenizer and model
    llm = LLM("t5-base", {"max_length": 50, "num_return_sequences": 1})

    # Create datasets
    train_dataset = TaskQuestionAnswerDataset(train_df, llm.tokenizer)
    val_dataset = TaskQuestionAnswerDataset(val_df, llm.tokenizer)

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=4)

    # Train the model
    train_model(llm.model, train_dataloader, val_dataloader)
    print("trained")
    # Save the model
    llm.model.save_pretrained('fine_tuned_t5_qa')
    llm.tokenizer.save_pretrained('fine_tuned_t5_qa')


def clean_text(text):
    # Remove special characters and excessive punctuation
    text = re.sub(r'[#%$@&*{}\[\]<>~]', '', text)
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    # Remove multiple punctuation
    text = re.sub(r'([!?,.])\1+', r'\1', text)
    # Clean up quotes
    text = re.sub(r'[\'"""]+', '"', text)
    return text.strip()

def generate_qa(model, tokenizer, task, environment, device):
    # Generate question
    task_input = f"generate question: {task} context: {environment}"
    inputs = tokenizer(task_input, return_tensors="pt", truncation=True).to(device)
    input_ids = inputs['input_ids'].to(device)

    gen_kwargs = {
        "max_length": 128,
        "min_length": 10,
        'num_beams' : 4,
        'no_repeat_ngram_size' : 2,
        "num_return_sequences": 1,
        "do_sample": True,
        "temperature": 0.9,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    question_ids = model.generate(input_ids, **gen_kwargs).to(device)
    # question = tokenizer.decode(question_ids[0], skip_special_tokens=True)
    question = []
    for sequence in question_ids:
        text = tokenizer.decode(
            sequence,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        question.append(text)

    # Generate answer variants
    answer_input = f"generate answer variants: {question}"
    inputs = tokenizer(answer_input, return_tensors="pt", truncation=True).to(device)
    input_ids = inputs['input_ids'].to(device)

    gen_kwargs = {
        "max_length": 128,
        "min_length": 10,
        'num_beams': 4,
        'no_repeat_ngram_size': 2,
        "num_return_sequences": 3,
        "do_sample": True,
        "temperature": 0.9,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    answer_ids = model.generate(input_ids, **gen_kwargs).to(device)
    answers = []
    for sequence in answer_ids:
        text = tokenizer.decode(
            sequence,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        answers.append(text)
    # answers = [tokenizer.decode(ids, skip_special_tokens=True) for ids in answer_ids]

    for i in range(len(answers)):
        answers[i] = clean_text(answers[i])
    return clean_text(question[0]), answers


if __name__ == "__main__":
    main() #fine-tuning model
    loaded_model = T5ForConditionalGeneration.from_pretrained('fine_tuned_t5_qa')
    loaded_tokenizer = T5Tokenizer.from_pretrained('fine_tuned_t5_qa')

    device = torch.device('mps')
    loaded_model = loaded_model.to(device)

    question, answers = generate_qa(loaded_model, loaded_tokenizer,
                                   'Kitchen Robot, please blend a bell pepper, a cucumber, a tomato, and a '
                                   'tablespoon of oil in the blender to make a fresh vegetable juice.',
                                   'kitchen towel, blender, porcelain cup, beer mug, ceramic mug, glass mug, '
                                   'plastic cup, paper cup, glass, bell pepper, cucumber, black pepper, tomato, '
                                   'sunflower oil, coconut oil, olive oil, black tea bags, green tea bags, '
                                   'fresh mozzarella package, cream cheese, cottage cheese, mozzarella sticks, '
                                   'cheddar cheese slices, vanilla yogurt cup, strawberry yogurt cup', 'mps')
    print(question)
    print(answers)
    # dataset = pd.read_csv('data/AmbiK_data.csv')[101:201]
    # results = pd.DataFrame(
    #     columns=['environment_full', "task", "is_amb", "orig_question", 'question', 'orig_all_variants',
    #              'answers', 'orig_answer']).astype({
    #     'environment_full': 'str',
    #     "task": 'str',
    #     "is_amb": 'bool',
    #     "orig_question": str,
    #     'orig_all_variants': str,
    #     'orig_answer': str,
    #     "question": 'str',
    #     'all_variants': 'str',
    #     'answer': 'str'
    # })
    #
    # new_rows = []
    # for i in range(len(dataset)):  # len(dataset)
    #     description = dataset.loc[i, 'environment_full']
    #     amb_task = dataset.loc[i, 'ambiguous_task']
    #     unamb_task = dataset.loc[i, 'unambiguous_indirect']
    #     question = dataset.loc[i, 'question']
    #     options = dataset.loc[i, 'variants']
    #     orig_answer = dataset.loc[i, 'answer']
    #
    #     question, answer = generate_qa(loaded_model, loaded_tokenizer, task, environment, 'mps')
    #
    #     new_rows.append({'environment_full': description, 'task': amb_task, 'is_amb': True, 'question': question,
    #                      'all_variants': options, 'answer': answer})
    #     new_rows.append({'environment_full': description, 'task': unamb_task, 'is_amb': False, 'question': question,
    #                      'all_variants': options, 'answer': answer})
    #
    # results = pd.concat([data_train_200, pd.DataFrame(new_rows)], ignore_index=True)
    # results.to_csv('data/results.csv')
    # # main()
