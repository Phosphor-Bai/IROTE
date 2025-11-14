import csv
import json
import os
import random
import scipy.stats as stats
from statistics import mean, stdev
import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd
import os
import random
import time
random.seed(42)
from abc import ABC, abstractmethod
import sys
PRESENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(PRESENT_DIR, os.pardir))
sys.path.append(os.path.join(BASE_DIR, 'models'))

class OtherContext:
    def __init__(self, data_name, num_k):
        # load data_dir
        data_dir = os.path.join(BASE_DIR, 'data')
        name2file = {
            'gsm8k': 'gsm8k.json',
            'human_eval': 'human_eval.json',
        }
        self.num_k = num_k
        self.data_name = data_name
        self.data = pd.read_json(os.path.join(data_dir, name2file[data_name]), lines=True)
    
    def get_message(self):
        index = random.sample(range(len(self.data)), self.num_k)
        tmp_data = self.data.iloc[index]
        questions = tmp_data['question'].tolist()
        answers = tmp_data['answer'].tolist()
        messages = []
        for question, answer in zip(questions, answers):
            messages.extend([
                {'role': 'user', 'content': question},
                {'role': 'assistant', 'content': answer}
            ])
        return messages
    
    def __str__(self):
        return f"{self.data_name}_{self.num_k}"

def single_query_llm(message, 
                        router,
                     model_name, max_tokens, temperature):
    return router.request_llm(message, model_name=model_name, max_length=max_tokens, temperature=temperature)

def batch_query_llms(messages, 
                     router,
                     model_name, max_tokens, temperature, batch_size=3):
    responses = []
    for batch in range(0, len(messages), batch_size):
        batch_messages = messages[batch:batch+batch_size] if batch + batch_size < len(messages) else messages[batch:]
        batch_responses = router.request_llm(batch_messages, model=model_name, max_length=max_tokens, temperature=temperature)
        responses.extend(batch_responses)
        time.sleep(3)
    return responses

# Handler: abstract class
class Handler(ABC):
    @abstractmethod
    def get_eval_results(self, **kwargs):
        pass
    
    def get_target_results(self, target_trait, **kwargs):
        pass

def get_questionnaire(questionnaire_name, base_dir):
    try:
        with open(os.path.join(base_dir, 'data/questionnaires.json')) as dataset:
            data = json.load(dataset)
    except FileNotFoundError:
        raise FileNotFoundError("The 'questionnaires.json' file does not exist.")

    # Matching by questionnaire_name in dataset
    questionnaire = None
    for item in data:
        if item["name"] == questionnaire_name:
            questionnaire = item

    if questionnaire is None:
        raise ValueError("Questionnaire not found.")

    return questionnaire


def generate_testdf(questionnaire, test_count, do_shuffle):
    questions_list = questionnaire["questions"] # get all questions
    df = pd.DataFrame()
    for shuffle_count in range(do_shuffle):
        question_indices = list(questions_list.keys())  # get the question indices

        # Shuffle the question indices
        if shuffle_count != 0:
            random.shuffle(question_indices)
        
        # Shuffle the questions order based on the shuffled indices
        questions = [f'{index}. {questions_list[question]}' for index, question in enumerate(question_indices, 1)]
        
        df[f'prompt-order-{shuffle_count}'] =  questions
        df[f'order-{shuffle_count}'] = question_indices
        for count in range(test_count):
            df[f'shuffle{shuffle_count}-test{count}'] = [''] * len(question_indices)
    return df


def convert_df(questionnaire, df):
    """
    Convert the DataFrame to a list of dictionaries.
    """
    test_data = []
    header = df.columns
    order_indices = []
    for index, column in enumerate(header):
        if column.startswith("order"):
            order_indices.append(index)
    for i in range(len(order_indices)):
        start = order_indices[i] + 1
        end = order_indices[i+1] - 1 if order_indices[i] != order_indices[-1] else len(header)
        for column_index in range(start, end):
            column_data = {}
            for row in df.iterrows():
                try: 
                    if "reverse" in questionnaire and int(row[1][start-1]) in questionnaire["reverse"]:
                        column_data[int(row[1][start-1])] = questionnaire["scale"] - int(row[1][column_index])
                    else:
                        column_data[int(row[1][start-1])] = int(row[1][column_index])
                except ValueError as e:
                    print(f'Column {column_index + 1} has error.', e)
                    sys.exit(1)
            test_data.append(column_data)
    return test_data


def compute_statistics(questionnaire, data_list):
    results = []
    
    for cat in questionnaire["categories"]:
        scores_list = []
        
        for data in data_list:
            scores = []
            for key in data:
                if key in cat["cat_questions"]:
                    scores.append(data[key])
            
            # Getting the computation mode (SUM or AVG)
            if questionnaire["compute_mode"] == "SUM":
                scores_list.append(sum(scores))
            else:
                scores_list.append(mean(scores))
        
        if len(scores_list) < 2:
            #raise ValueError("The test file should have at least 2 test cases.")
            results.append((mean(scores_list), 0, len(scores_list)))
            continue
        
        results.append((mean(scores_list), stdev(scores_list), len(scores_list)))
        
    return results


import re
def contains_ai(sentence):
		"""
		Function to check if a sentence contains the isolated keyword "AI" with word boundaries,
		including cases where "AI" might be followed by punctuation like "AI." but not part of another word.
		"""
		# Regular expression pattern to match isolated "AI" with word boundaries, including following punctuation
		ai_pattern = re.compile(r'\bAI\b\.?')
		
		# Search for the pattern in the sentence
		match = ai_pattern.search(sentence)
		
		# Return True if a match is found, False otherwise
		return bool(match) or ('language model' in sentence.lower())



    