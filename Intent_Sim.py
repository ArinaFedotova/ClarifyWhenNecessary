import numpy as np
import pandas as pd
from scipy.stats import entropy
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import os

from utils.parse_config import parse_config
from prompts import examples_generation, answer_generation

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from utils.llm import LLM
from utils.nli import NLI


class Intent_Sim:
    def __init__(self, config):
        config_mapping = {
            'examples_generation': ['model', 'generation_kwargs'],
            'certainty_model': ['model', 'generation_kwargs'],
        }
        self.config = config
        self.examples_generation = config['examples_generation']
        self.certainty_model = config['certainty_model']

    def greedy_sample(self, llm, input_text):
        response = llm.generate(input_text)
        # self, prompt, return_logits = False, max_length = 50, num_beams = 4, no_repeat_ngram_size = 2,
        # num_return_sequences = 2, do_sample = True, temperature = 0.9
        return response[0]

    def temp_sample(self, llm, input_text, temperature=0.5):
        """Генерация ответов с температурой"""
        print("response")
        print(input_text)
        response = llm.generate(input_text[0])
        print(response)
        answer, logits = llm.generate(input_text, return_logits=True)
        llm.filter_logits(logits[0][1], words = answer)
        print(answer)
        return response[0]

    def dfs(self, graph, node, visited=None):
        """Поиск в глубину для нахождения компонент связности"""
        if visited is None:
            visited = set()
        visited.add(node)
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                self.dfs(graph, neighbor, visited)
        return visited

    def simularity (self, nli, sentence1, sentence2):
        return nli.check_entailment(sentence1, sentence2)

    def calculate_uncertainty(self, user_input, temperature=0.5, simulation_count=10):
        llm = LLM(self.examples_generation['model'],
                  self.examples_generation['generation_kwargs'])
        nli = NLI(self.certainty_model['model'],
                  self.certainty_model['generation_kwargs'])
        print("generated")
        """Основной метод для расчета неопределенности"""
        # 1. Генерация уточняющего вопроса
        clarifying_question = self.greedy_sample(llm, user_input)
        print("1")
        # 2. Симуляция различных намерений пользователя
        simulated_responses = []
        for _ in range(simulation_count):

            response = self.temp_sample(llm,[user_input, clarifying_question], temperature)
            simulated_responses.append(response)
        print("2")
        # 3. Построение графа эквивалентности
        graph = {}
        for i in range(simulation_count - 1):
            for j in range(i + 1, simulation_count):
                qa_pair_i = f"{clarifying_question} {simulated_responses[i]}"
                qa_pair_j = f"{clarifying_question} {simulated_responses[j]}"

                # Проверка энтейлмента в обоих направлениях
                if (self.simularity(nli, qa_pair_i, qa_pair_j) or self.simularity(nli, qa_pair_j, qa_pair_i)):
                    if i not in graph:
                        graph[i] = []
                    if j not in graph:
                        graph[j] = []
                    graph[i].append(j)
                    graph[j].append(i)

        # 4. Нахождение компонент связности (различных намерений)
        components = []
        visited_nodes = set()
        for i in range(simulation_count):
            if i not in visited_nodes:
                component = self.dfs(graph, i)
                components.append(component)
                visited_nodes.update(component)

        # 5. Расчет вероятностей и энтропии
        probabilities = [len(c) / simulation_count for c in components]
        uncertainty = entropy(probabilities)

        return uncertainty, components, simulated_responses

if __name__ == "__main__":
    configs = parse_config("utils/CWN.yaml", use_args=True)
    gen_model = configs['examples_generation']['model']
    if "/" in gen_model:
        gen_model = gen_model.split("/")[1]

    nli_model = configs['certainty_model']['model']
    if "/" in nli_model:
        nli_model = nli_model.split("/")[1]
    exp_res_dir = f"./{gen_model}_{nli_model}"
    # os.makedirs(exp_res_dir, exist_ok=True)

    # Initialize Intent_Sim
    intent_sim = Intent_Sim(configs)

    # Read the DataFrame
    df = pd.read_csv("data/AmbiK_data.csv",
                     nrows=100,
                     usecols=["environment_full", "unambiguous_indirect",
                              "ambiguous_task", "plan_for_amb_task"])

    # Get the ambiguous task text from row 1
    ambiguous_task = df.loc[1, "ambiguous_task"]

    # Calculate uncertainty
    uncertainty, components, simulated_responses = intent_sim.calculate_uncertainty(
        user_input=ambiguous_task,
        temperature=0.5,
        simulation_count=10
    )

    # Print results
    print(f"Uncertainty: {uncertainty}")
    print(f"Components: {components}")
    print(f"Simulated responses: {simulated_responses}")