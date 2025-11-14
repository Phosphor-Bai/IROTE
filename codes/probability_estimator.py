from concurrent.futures import ThreadPoolExecutor
import re
import numpy as np
from models.llm_interface import LLMFactory
from scipy.sparse import coo_matrix

"""
Use prompting to let LLM output a score between 0 and 10 to approximate the probability, representing the likelihood of text t1 appearing given text t2, i.e., P(t1|t2).
"""

# Evaluate P(text1 | text2)
# 1. straghtforward
def get_eval_prompt1(texta:str, textb:str, inverse=False) -> str:
    """
    Straightforward way to evaluate P(text1 | text2)
    """
    pos_a = "1" 
    pos_b = "2" 
    if inverse:
        pos_a, pos_b = pos_b, pos_a
        texta, textb = textb, texta
    return f"""In the context of language modeling, we want to estimate the conditional probability P(Text 1 | Text 2). Please provide a score from 0 to 10 to represent this probability, where 0 means P(Text 1 | Text 2) is essentially zero, and 10 means P(Text 1 | Text 2) is very close to one. Consider the following texts:

[Text {pos_a}]: 
{texta}

[Text {pos_b}]: 
{textb}

Score (representing P(Text 1 | Text 2)): """

# 2. 基于文本蕴含（Textual Entailment）
def get_eval_prompt2(texta:str, textb:str, inverse=False) -> str:
    pos_a = "1"
    pos_b = "2"
    if inverse:
        pos_a, pos_b = pos_b, pos_a
        texta, textb = textb, texta
    return f"""On a scale from 0 to 10, where 0 means Text 1 provides absolutely no evidence for Text 2, and 10 means Text 1 completely and undeniably entails Text 2, how strongly does Text 1 support or imply Text 2?

[Text {pos_a}]: 
{texta}

[Text {pos_b}]: 
{textb}
Score: """

# 3. 基于relatedness
def get_eval_prompt3(texta:str, textb:str, inverse=False) -> str:
    pos_a = "1"
    pos_b = "2"
    if inverse:
        pos_a, pos_b = pos_b, pos_a
        texta, textb = textb, texta
    return f"""On a scale from 0 to 10, where 0 means Text 1 is completely unrelated to Text 2, and 10 means Text 1 is almost identical to Text 2, how related are Text 1 and Text 2?

[Text {pos_a}]: 
{texta}

[Text {pos_b}]: 
{textb}
Score: """

# 4. 基于文本生成（Text Generation）
def get_eval_prompt4(texta:str, textb:str, inverse=False) -> str:
    pos_a = "1"
    pos_b = "2"
    if inverse:
        pos_a, pos_b = pos_b, pos_a
        texta, textb = textb, texta
    return f"""On a scale from 0 to 10, where 0 means Text 1 is completely unrelated to Text 2, and 10 means Text 1 is almost identical to Text 2, how likely is Text 1 to be generated from Text 2?

[Text {pos_a}]: 
{texta}

[Text {pos_b}]: 
{textb}
Score: """


def extract_scores(response:str) -> float:
    """
    Extract the score from the response
    """
    try:
        return float(response.strip())
    except:
        digit = re.findall(r"\d+", response)
        if digit:
            return float(digit[0])
        else:
            return None


class ProbabilityEstimator:
    def __init__(self, prompt_type_names="1,2,4"):
        prompt_types = [int(i)-1 for i in prompt_type_names.split(",")]
        all_eval_types = [get_eval_prompt1, get_eval_prompt2, get_eval_prompt3, get_eval_prompt4]
        self.eval_funcs = [all_eval_types[i] for i in prompt_types]

    def get_score(self, texta:str, textb:str, model_name) -> float:
        """
        Request for a score to estimate P(texta | textb)
        """
        messages = []
        for is_inverse in [False, True]:
            for eval_func in self.eval_funcs:
                prompt = eval_func(texta, textb, is_inverse)
                # print(prompt)
                # print('*'*20)
                messages.append([
                    {"role": "user", "content": prompt},
                ])
        # get response
        responses = LLMFactory.gather_multiple_messages(messages, model_name=model_name, temperature=0.001, max_tokens=100)
        # extract scores
        scores = [extract_scores(response) for response in responses if extract_scores(response) is not None]
        return np.mean(scores) * 0.1
                


def calculate_reflection_score(pair, model_name='GPT-4o'):
    """
    Calculate the score of reflection1 given reflection2
    """
    reflection1, reflection2 = pair
    prob_estimator = ProbabilityEstimator(prompt_type_names="1,2,4")
    str1 = reflection1.get_integrated_strs()
    str2 = reflection2.get_integrated_strs()
    return prob_estimator.get_score(str1, str2, model_name)
    


def request_for_similarity_score(
        new_reflections, 
        other_reflections, 
        score_model_name='GPT-4o',
        batch_size=1,
        parallel=False
):
    flatten_pairs_e_c, flattern_idxs_e_c = [], []
    flatten_pairs_c_e, flattern_idxs_c_e = [], []
    for i, new_reflection in enumerate(new_reflections):
        for j, prev_reflection in enumerate(other_reflections):
            flatten_pairs_e_c.append((new_reflection, prev_reflection))
            flattern_idxs_e_c.append((i, j))
            flatten_pairs_c_e.append((prev_reflection, new_reflection))
            flattern_idxs_c_e.append((j, i))
    # Use ThreadPoolExecutor to parallelize the requests
    if parallel:
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            prob_e_c = list(executor.map(calculate_reflection_score, flatten_pairs_e_c, [score_model_name]*len(flatten_pairs_e_c)))
            prob_c_e = list(executor.map(calculate_reflection_score, flatten_pairs_c_e, [score_model_name]*len(flatten_pairs_c_e)))
    else:
        prob_e_c = [calculate_reflection_score(pair, score_model_name) for pair in flatten_pairs_e_c]
        prob_c_e = [calculate_reflection_score(pair, score_model_name) for pair in flatten_pairs_c_e]
        
    idxs1, idxs2 = zip(*flattern_idxs_e_c)
    prob_e_c = coo_matrix((prob_e_c, (idxs1, idxs2)), shape=(len(new_reflections), len(other_reflections))).toarray()
    idxs1, idxs2 = zip(*flattern_idxs_c_e)
    prob_c_e = coo_matrix((prob_c_e, (idxs1, idxs2)), shape=(len(other_reflections), len(new_reflections))).toarray()
    return prob_e_c, prob_c_e

def calculate_term_1(new_reflections, prev_reflections, batch_size=5, score_model_name='GPT-4o'):
    """
    new_reflections: list of target reflection: e
    prev_reflections: list of previous reflections: ck
    """
    prob_e_c, prob_c_e = request_for_similarity_score(new_reflections, prev_reflections, 
                                                      batch_size=batch_size, score_model_name=score_model_name)
    log_prob_c_e = np.log(prob_c_e)
    # sum(p(e|e_k) * log(p(e_k|e)))
    term1_res = np.zeros(len(new_reflections))
    for e_idx in range(len(new_reflections)):
        term1_res[e_idx] = np.sum(prob_e_c[e_idx] * log_prob_c_e[:, e_idx])
    return term1_res

def calculate_term_2(
        new_reflections,
        all_reflections,
        batch_size=1,
        score_model_name='GPT-4o'
    ):
    """
    new_reflections: list of target reflection: e
    all_reflections: list of all reflections: E
    """
    prob_e_c, prob_c_e = request_for_similarity_score(new_reflections, [all_reflections], 
                                                      batch_size=batch_size, score_model_name=score_model_name)
    log_prob_c_e = np.log(prob_c_e)
    term2_res = np.zeros(len(new_reflections))

    for e_idx in range(len(new_reflections)):
        # p(e|E) [log(p(E|e)) - other average log(p(E|e))]
        positives = log_prob_c_e[:, e_idx]
        negatives = (np.sum(log_prob_c_e, axis=1) - positives) / (len(new_reflections) - 1)
        term2_res[e_idx] = prob_e_c[e_idx] * (positives - negatives)
    return term2_res    
            

def calculate_compactness(new_reflections, 
                          prev_reflections,
                          all_reflection, 
                            batch_size=1,
                            score_model_name='GPT-4o'
                          ):
    """
    new_reflections: list of target reflection: e
    prev_reflections: list of previous reflections: e_k
    all_reflection: list of all reflections: E
    """
    # term 1. e, ek: cross-entropy 
    term1 = calculate_term_1(new_reflections, prev_reflections
                                , batch_size=batch_size, score_model_name=score_model_name)
    # term 2. e, E contrastive score
    term2 = calculate_term_2(new_reflections, all_reflection
                                , batch_size=batch_size, score_model_name=score_model_name)
    compactness_score = term1 - term2

    # update the reflections
    for i in range(len(new_reflections)):
        new_reflections[i].update_compactness(
            compactness=compactness_score[i],
            term1=term1[i],
            term2=term2[i]
        )
    token_count = LLMFactory.print_token_count(model_name=score_model_name)
    print(f"Used Tokens: {token_count}M")
    return compactness_score


if __name__=="__main__":
    # test
    texta = "The dog has been sleeping for hours."
    textb = "The dog has been sleeping on the bottom of the bed for hours."
    model_name = "GPT-4o"
    estimator = ProbabilityEstimator(prompt_type_names="1,2,4")
    score = estimator.get_score(texta, textb, model_name)
    print(score)
    