from functools import reduce
import json
import os
import pandas as pd
import time
import re
from tqdm import tqdm
from .utils import compute_statistics, get_questionnaire, generate_testdf, convert_df
from .utils import Handler, BASE_DIR, batch_query_llms, single_query_llm

# from models.router import ModelRouter

def extract_digits(string):
    # search digits
    groups = re.search(r'\d+', string)
    # may have multiple digits
    if groups:
        # return the first one
        return int(groups.group())
    else:
        return None

def get_digit_text(string, dilimiters=[':', '：']):
    string = string.strip()
    split_dilimiter = None
    for dilimiter in dilimiters:
        if dilimiter in string:
            split_dilimiter = dilimiter
            break
    if split_dilimiter is None:
        return string
    try:
        part1, part2 = string.split(split_dilimiter)
        score1, score2 = extract_digits(part1), extract_digits(part2)
        if score1 is not None and score2 is not None:
            return part2
        elif score1 is None:
            return part2
        elif score2 is None:
            return part1
        else:
            return string
    except:
        return string

def convert_results(result, column_header):
    # extract from all text
    result = result.replace(".", "").strip()
    try:
        result_list = []
        for element in result.split('\n'):
            if element.strip() == '':
                continue
            element = get_digit_text(element)
            digit = extract_digits(element)
            if digit is not None:
                result_list.append(digit)
            else:
                raise ValueError("No digit found.")
    except:
        print(f"Unable to capture the responses on {column_header}.")
        return [0] * len(result.split('\n'))
    return result_list

def convert_results_seperately(response_list, column_header):
    try:
        result_list = []
        for element in response_list:
            element = get_digit_text(element)
            digit = extract_digits(element)
            if digit is not None:
                result_list.append(digit)
            else:
                raise ValueError("No digit found.")
    except:
        print(f"Unable to capture the responses on {column_header}.")
        print(response_list)
        return [0] * len(response_list)
    return result_list
    

    
def get_single_query(inner_setting: str, 
                     questionnaire_instruction: str,
                     questions_string, 
                     shot_str=None,
                     in_system=False,
                     insert_history: list=None
                     )->list:
    """
    Get next query input
    """
    if shot_str is not None:
        inner_setting = shot_str + inner_setting
    if in_system:
        if insert_history is not None:
            return [{"role": "system", "content": inner_setting}] + insert_history + [{"role": "user", "content": questionnaire_instruction + '\n' + questions_string}]
        else:
            return [
                {"role": "system", "content": inner_setting},
                {"role": "user", "content": questionnaire_instruction + '\n' + questions_string}
            ]
    else:
        if insert_history is not None:
            return insert_history + [
                {"role": "user", "content":  inner_setting+ '\n' + questionnaire_instruction + '\n' + questions_string}
            ]
        else:
            return [
                {"role": "user", "content":  inner_setting+ '\n' + questionnaire_instruction + '\n' + questions_string}
            ]
    

class SurveyHandler(Handler):
    def __init__(self,
                 router,
                 test_questionnaire,
                 shuffle_count=2,
                 test_count=1,
                 in_system=False,
                 batch_size=1,
                 crowd_analysis=False,
                 temperature=0.01,
                 other_context=None,
                 condition_mode=False,
                 save_dir=None
    ):
        self.router = router
        self.shuffle_count = shuffle_count
        self.test_count = test_count
        self.in_system = in_system
        self.batch_size = batch_size
        self.crowd_analysis = crowd_analysis
        self.temperature = temperature
        self.save_dir = save_dir
        
        self.questionnaire_list = ['BFI','BFI2' 'MFQ-1', 'MFQ-2', 'SVS', 'PVQ-RR', 'PVQ21'
                          ] if test_questionnaire == 'ALL' else test_questionnaire.split(',')
        self.other_context = other_context
        self.condition_mode = condition_mode

    def get_target_results(self,
                           target_trait:str,
                           model_name:str,
                           shot_str=None,
                           max_tokens=1024,
                           temperature=0.01,
                           ):
        res_dict_list = []
        for questionnaire_name in self.questionnaire_list:
            questionnaire = get_questionnaire(questionnaire_name, BASE_DIR)
            questions_map = questionnaire['questions']
            categories = {cat['cat_name'].lower(): cat['cat_questions'] for cat in questionnaire['categories']}
            assert target_trait in categories, f"{target_trait} is not in the categories of {questionnaire_name}."
            trait_idxes = categories[target_trait]
            questions_list = [questions_map[str(q)] for q in trait_idxes]

            insert_history = None

            # template settings
            inner_setting = questionnaire["inner_setting_single"] if not self.condition_mode else questionnaire["inner_setting"]
            questionnaire_instruction = questionnaire["prompt_single"] if not self.condition_mode else questionnaire["prompt"]
        
            result_list = self.query_seperatly(
                            column_header="",
                            inner_setting=inner_setting,
                            questionnaire_instruction=questionnaire_instruction,
                            model_name=model_name,
                            questions_list=questions_list,
                            shot_str=shot_str,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            insert_history=insert_history
                        )
            assert len(trait_idxes) == len(result_list) == len(questions_list), f"Lengths of trait_idxes, result_list, and questions_list are not equal."
            tmp_res = []
            reverse_list = questionnaire.get('reverse', None)
            div_scale = questionnaire['scale'] - 1
            for q_idx, q, res in zip(trait_idxes, questions_list, result_list):
                if reverse_list is not None and q_idx in reverse_list:
                    res = questionnaire['scale'] - res
                else:
                    res = res
                tmp_res.append(res / div_scale * 10)
            res_dict_list.append({target_trait: sum(tmp_res) / len(tmp_res)})
        return SurveyHandler.merge_results(res_dict_list, avg_mode='algorithmic_mean')

    def get_eval_results(self,
                            model_name:str,
                            shot_str=None,
                            max_tokens=1024,
                            temperature=0.01,
                            save_dir=None
                            ):
        """
        Original Psychobench evaluation
        """
        res_dict_list, scale_dict_list, gathered_res_list = [], [], []
        for questionnaire_name in self.questionnaire_list:
            questionnaire = get_questionnaire(questionnaire_name, BASE_DIR)
            df = generate_testdf(questionnaire,
                              test_count=self.test_count,
                              do_shuffle=self.shuffle_count)
            
            shuffle_count = 0
            prompt_columns_idxs = [i for i, col in enumerate(df.columns) if col.lower().startswith("prompt")]
            for questions_column_index in prompt_columns_idxs:
                questions_list = df.iloc[:, questions_column_index].astype(str)
                # Handle insert_history for long-context or other robustness test
                insert_history = None
                if self.other_context is not None:
                    insert_history = self.other_context.get_message()
                for k in range(self.test_count):
                    # query the responses
                    column_header = f'shuffle{shuffle_count}-test{k}'
                    if  self.condition_mode:
                        result_list = self.query_conditionally(
                            column_header=column_header,
                            inner_setting=questionnaire["inner_setting"],
                            questionnaire_instruction=questionnaire["prompt"],
                            model_name=model_name,
                            questions_list=questions_list,
                            shot_str=shot_str,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            insert_history=insert_history
                        )
                    else:
                        result_list = self.query_seperatly(
                            column_header=column_header,
                            inner_setting=questionnaire["inner_setting"],
                            questionnaire_instruction=questionnaire["prompt"],
                            model_name=model_name,
                            questions_list=questions_list,
                            shot_str=shot_str,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            insert_history=insert_history
                        )
                    try:
                        if column_header in df.columns:
                            df[column_header] = result_list
                    except Exception as e:
                        print(f"Unable to capture the responses on {column_header}.", e)
                shuffle_count += 1
            # analysis
            converted_data_list = convert_df(questionnaire, df)
            test_results = compute_statistics(questionnaire, converted_data_list)
            gathered_json_res = {
                'name': questionnaire['name'],
                'dimensions': [cat['cat_name'].lower() for cat in questionnaire['categories']],
                'mean': [res[0] for res in test_results],
                'std': [res[1] for res in test_results],
                'n': test_results[0][2]
            }
            
            tmp_test_res = {k:v for k, v in zip(gathered_json_res['dimensions'], gathered_json_res['mean'])}
            scale_num = questionnaire['scale'] - 1
            scale_test_res = {k:v/scale_num*10 for k, v in tmp_test_res.items()}
            res_dict_list.append(tmp_test_res)
            scale_dict_list.append(scale_test_res)
            gathered_res_list.append(gathered_json_res)
        
        # Save the results
        save_dict = {
            'questionnaire_list': self.questionnaire_list,
            'results': res_dict_list,
            'scale_results': scale_dict_list,
            'gathered_results': gathered_res_list,
        }
        if save_dir is not None:
            save_name = '_'.join(self.questionnaire_list)+ "_" + str(self.other_context) + '_results.json' if self.other_context is not None else '_'.join(self.questionnaire_list) + '_results.json'
            save_path = os.path.join(save_dir, save_name)
            with open(save_path, 'w') as f:
                json.dump(save_dict, f)
        return SurveyHandler.merge_results(scale_dict_list, avg_mode='algorithmic_mean')
    
    @classmethod
    def merge_results(cls, results_list, avg_mode='algorithmic_mean'):
        """
        Merge the results from different questionnaires
        """
        res_dict = {}
        for res in results_list:
                for k, v in res.items():
                    if k in res_dict:
                        res_dict[k].append(v)
                    else:
                        res_dict[k] = [v]
        if avg_mode == 'algorithmic_mean':
            for k, v in res_dict.items():
                res_dict[k] = sum(v) / len(v)
        elif avg_mode == 'harmonic_mean':
            for k, v in res_dict.items():
                res_dict[k] = len(v) / sum([1 / val for val in v])
        elif avg_mode == 'geometric_mean':
            for k, v in res_dict.items():
                res_dict[k] = (reduce(lambda x, y: x*y, v))**(1/len(v))
        # rescale to 0 - 10
        return res_dict

    def query_conditionally(self,
                            column_header:str,
                            inner_setting:str,
                            questionnaire_instruction:str,
                            model_name:str,
                            questions_list:list,
                            shot_str=None,
                            max_tokens=1024,
                            temperature=0.01,
                            insert_history=None
                            ):
        result_string_list = []
        previous_records = []
        if insert_history is not None:
            previous_records = insert_history
        
        for batch in range(0, len(questions_list), self.batch_size):
            result = ''

            batch_questions = questions_list[batch:batch+self.batch_size] if batch + self.batch_size < len(questions_list) else questions_list[batch:]
            if len(batch_questions) == 1:
                questions_string = batch_questions[0]
            else:
                questions_string = '\n'.join([f"{i+1}. {q.split('.')[1]}" for i, q in enumerate(batch_questions)])

            inputs = previous_records + get_single_query(
                inner_setting=inner_setting,
                questionnaire_instruction=questionnaire_instruction,
                questions_string=questions_string,
                shot_str=shot_str,
                in_system=self.in_system
            )
            result = single_query_llm(message=inputs, 
                                      router=self.router,
                                        model_name=model_name,
                                        max_tokens=max_tokens,
                                        temperature=temperature
                                    )
            previous_records.append({"role": "user", "content": questionnaire_instruction + '\n' + questions_string})
            previous_records.append({"role": "assistant", "content": result})
        
            result_string_list.append(result.strip())
        
        result_string = '\n'.join(result_string_list)
        
        result_list = convert_results(result_string, column_header)
        if result_list is None:
            return [0] * len(questions_list)
        return result_list


    def query_seperatly(self,
                        column_header:str,
                        inner_setting:str,
                        questionnaire_instruction:str,
                        model_name:str,
                        questions_list:list,
                        shot_str=None,
                        max_tokens=1024,
                        temperature=0.01,
                        insert_history=None,
                        ):
        
        messages = []
        for questions_string in questions_list:
            messages.append(get_single_query(
                inner_setting=inner_setting,
                questionnaire_instruction=questionnaire_instruction,
                questions_string=questions_string,
                shot_str=shot_str,
                in_system=self.in_system,
                insert_history=insert_history
            ))
        
        responses = batch_query_llms(messages, 
                                    router=self.router,
                                     model_name=model_name, max_tokens=max_tokens, temperature=temperature)
        result_list = convert_results_seperately(responses, column_header)
        return result_list

