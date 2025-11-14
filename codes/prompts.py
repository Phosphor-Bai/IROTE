import re
from typing import List, Dict, Any, Tuple, Union
from models.llm_interface import LLMFactory

def get_single_policy(case_shots: List[str],  policy_index: int, score: int=None) -> str:
    case_shots = "\n".join([f"{i+1}. {shot_str}" for i, shot_str in enumerate(case_shots)])
    if score is None:
        return f"""[POLICY] - {policy_index}
{case_shots}"""
    else:
        return f"""[POLICY] - {policy_index}
{case_shots}
[SCORE]
{score}"""

def organize_policies(shot_strs_list: List[Tuple[List[str], int]]) -> str:
    # case_shots = "\n".join([f"{i+1}. {shot_str}" for i, shot_str in enumerate(shot_strs)])
    policy_strs = []
    for i, (case_shots, score) in enumerate(shot_strs_list):
        policy_strs.append(get_single_policy(case_shots, 
                                             score=score, 
                                             policy_index=i+1))
    return "\n\n".join(policy_strs)


def get_optimize_policy_prompt(shot_strs_list: List[Tuple[List[str], int]]=None, 
                               num_words:int=None, 
                               use_cot:bool=False, 
                               is_step2:bool=False, 
                               task_description:str=None) -> str:
    """
    Used for generating the prompt for the optimization task
    """
    # case_shots=None
    if is_step2:
        assert use_cot and num_words is not None
        return f"""Now, based on the above analysis, organize a new policy. Remember, the new policy should strictly follow the policy format, and it should not exceed {num_words} words in total."""
    else:
        assert shot_strs_list is not None and num_words is not None
        case_shots = organize_policies(shot_strs_list)
        if task_description is not None:
            optimize_policy_prompt = f"""# BACKGROUND
We are trying to search for the best control policy for an agent that completes specific tasks. Here is the task description:
{task_description}


A policy is multiple lines where each line contains a reflection and an action comparison(<text of reflection>, e.g.: <text of action comparison>)
Here is an example of a policy(just for reference):

[POLICY] - <policy index>
1. I am curious and eager to learn new things, e.g.: I often find myself researching topics that I'm not familiar with, just for the sake of expanding my knowledge.
2. I enjoy exploring new cultures and ways of life, e.g.: I would rather travel to a country I've never been to before than revisit a familiar destination.
3. ...

# INSTRUCTION
Now, we need to optimize for a new policy based on a set of reflections and their scores(higher the score, better the policy, max score is 10)
The policies and their scores are given in the following format:

[POLICY] - 1
1. <reflection 1>, e.g.: <action comparison 1>
2 ...
[SCORE]
<score 1>

[POLICY] - 2
...(repeat the same format for other policies)

So, you need to optimize for a new policy based on the given set of policies and their scores. Both analysis, exploration, and summarization are quite important in optimizing for the new policy.

# CASE TO BE OPTIMIZED
{case_shots}

Now please optimize for a new policy. Remember, the new policy any number of lines but it should not exceed {num_words} words in total. """
        else:
            optimize_policy_prompt = f"""# BACKGROUND
We are trying to search for the best control policy for an agent that completes specific tasks.
A policy is multiple lines where each line contains a reflection and an action comparison(<text of reflection>, e.g.: <text of action comparison>)
Here is an example of a policy(just for reference):

[POLICY] - <policy index>
1. I am curious and eager to learn new things, e.g.: I often find myself researching topics that I'm not familiar with, just for the sake of expanding my knowledge.
2. I enjoy exploring new cultures and ways of life, e.g.: I would rather travel to a country I've never been to before than revisit a familiar destination.
3. ...

# INSTRUCTION
Now, we need to optimize for a new policy based on a set of reflections and their scores(higher the score, better the policy, max score is 10)
The policies and their scores are given in the following format:

[POLICY] - 1
1. <reflection 1>, e.g.: <action comparison 1>
2 ...
[SCORE]
<score 1>

[POLICY] - 2
...(repeat the same format for other policies)

So, you need to optimize for a new policy based on the given set of policies and their scores. Both analysis, exploration, and summarization are quite important in optimizing for the new policy.

# CASE TO BE OPTIMIZED
{case_shots}

Now please optimize for a new policy. Remember, the new policy any number of lines but it should not exceed {num_words} words in total. """
    if use_cot:
        optimize_policy_prompt += "Let's think step by step, "
        return optimize_policy_prompt
    else:
        return optimize_policy_prompt + "And it should strictly follow the policy([POLICY]) format: "


def get_summarization_policy_prompt(shot_str_list: List[str]=None, 
                                num_words:int=None, 
                                task_description:str=None) -> str:
    """
    Used for generating the prompt for the summarization task
    """
    case_shots = get_single_policy(shot_str_list, policy_index=1)
    if task_description is not None:
        summarize_policy_prompt = f"""# BACKGROUND
We are trying to search for the best control policy for an agent that completes specific tasks. Here is the task description:
{task_description}

A policy is multiple lines where each line contains a reflection and an action comparison(<text of reflection>, e.g.: <text of action comparison>)
# INSTRUCTION
Now, we need to summarize the given policy. The policies and their scores are given in the following format:

[POLICY] - 1
1. <reflection 1>, e.g.: <action comparison 1>
2 ...

# CASE TO BE SUMMARIZED
{case_shots}
So, you need to summarize the given policy. Your summary should be concise and capture the essence of the policy. The summary should not exceed {num_words} words in total with the same format as the policy: """
    else:
        summarize_policy_prompt = f"""# BACKGROUND
We are trying to search for the best control policy for an agent that completes specific tasks. 
A policy is multiple lines where each line contains a reflection and an action comparison(<text of reflection>, e.g.: <text of action comparison>)
# INSTRUCTION
Now, we need to summarize the given policy. The policies and their scores are given in the following format:

[POLICY] - 1
1. <reflection 1>, e.g.: <action comparison 1>
2 ...

# CASE TO BE SUMMARIZED
{case_shots}
So, you need to summarize the given policy. Your summary should be concise and capture the essence of the policy with no reflection number constrain. The total summary should not exceed {num_words} words in total with the same format as the policy: """
    return summarize_policy_prompt

def extract_reflection(response):
        # extract 1. xxxxx \n 2. xxxxx \n 3. xxxxx
        return [text.strip() for text in re.findall(r'\d+\.\s(.+)', response)]

def summary_reflections(router,
                        shot_strs_list: List[List[str]], 
                            model:str, 
                            words_limit:int, 
                            task_description:str=None,
                            max_tokens:int=2048,
                            temperature:float=1.0
                            ) -> List[Dict[str, Any]]:

        query_messages = []
        for i, shot_strs in enumerate(shot_strs_list):
            query_messages.append([
                {"role": "user", "content": get_summarization_policy_prompt(shot_str_list=shot_strs, 
                                                                            num_words=words_limit,
                                                                            task_description=task_description)}
            ])
        # LLMFactory.gather_multiple_messages(query_messages, model_name=model, temperature=temperature, max_tokens=max_tokens)
        responses = router.request_llm(conversations=query_messages, model=model, max_length=max_tokens, temperature=temperature)
        summaries = [extract_reflection(response) for response in responses]
        return summaries


def optimize_reflections(router,
                            shot_strs_list: List[Tuple[List[str], int]], 
                            model:str, 
                            words_limit:int, 
                            task_description:str=None,
                            max_tokens:int=2048,
                            temperature:float=1.0
                            ) -> List[Dict[str, Any]]:

        query_messages = []
        for i, (shot_strs, score) in enumerate(shot_strs_list):
            query_messages.append([
                {"role": "user", "content": get_optimize_policy_prompt(shot_strs_list=[(shot_strs, score)], 
                                                                        num_words=words_limit,
                                                                        task_description=task_description)}
            ])
        # LLMFactory.gather_multiple_messages(query_messages, model_name=model, temperature=temperature, max_tokens=max_tokens)
        responses = router.request_llm(conversations=query_messages, model=model, max_length=max_tokens, temperature=temperature)
        res = [extract_reflection(response) for response in responses]
        return res

def cot_optimize_reflections(
                            router,
                            shot_strs_list: List[Tuple[List[str], int]], 
                            model:str, 
                            words_limit:int, 
                            task_description:str=None,
                            max_tokens:int=2048,
                            temperature:float=1.0
                            ) -> List[Dict[str, Any]]:
        # step 1
        print("step 1")
        query_messages = []
        for i, (shot_strs, score) in enumerate(shot_strs_list):
            query_messages.append([
                {"role": "user", "content": get_optimize_policy_prompt(shot_strs_list=[(shot_strs, score)], 
                                                                        num_words=words_limit,
                                                                        task_description=task_description, 
                                                                        use_cot=True,
                                                                        is_step2=False)}
            ])
        # LLMFactory.gather_multiple_messages(query_messages, model_name=model, temperature=temperature, max_tokens=max_tokens)
        cot_responses = router.request_llm(conversations=query_messages, model=model, max_length=max_tokens, temperature=temperature)
        # step 2
        print("step 2")
        for i in range(len(query_messages)):
            query_messages[i].append(
                 {"role": "assistant", "content": cot_responses[i]}
                 )
            query_messages[i].append(
                {"role": "user", "content": get_optimize_policy_prompt(num_words=words_limit, 
                                                                        use_cot=True, 
                                                                        is_step2=True)}
            )
        # LLMFactory.gather_multiple_messages(query_messages, model_name=model, temperature=temperature, max_tokens=max_tokens)
        responses = router.request_llm(conversations=query_messages, model=model, max_length=max_tokens, temperature=temperature)
        res = [extract_reflection(response) for response in responses]
        return res

if __name__=='__main__':
    shot_strs_list = [
        (["I am curious and eager to learn new things, e.g.: I often find myself researching topics that I'm not familiar with, just for the sake of expanding my knowledge.",
          "I enjoy exploring new cultures and ways of life, e.g.: I would rather travel to a country I've never been to before than revisit a familiar destination.",
          "I am a good listener, e.g.: I always try to understand the other person's perspective before responding."], 7),
        (["I am curious and eager to learn new things, e.g.: I often find myself researching topics that I'm not familiar with, just for the sake of expanding my knowledge.",
          "I enjoy exploring new cultures and ways of life, e.g.: I would rather travel to a country I've never been to before than revisit a familiar destination.",
          "I am a good listener, e.g.: I always try to understand the other person's perspective before responding."], 8),
        (["I am curious and eager to learn new things, e.g.: I often find myself researching topics that I'm not familiar with, just for the sake of expanding my knowledge.",
          "I enjoy exploring new cultures and ways of life, e.g.: I would rather travel to a country I've never been to before than revisit a familiar destination.",
          "I am a good listener, e.g.: I always try to understand the other person's perspective before responding."], 5)
    ]

    task_description = "**Task for Science Quiz**: The AI agent needs to answer a set of science questions. "


    from models.llm_interface import LLMFactory
    prompt = get_summarization_policy_prompt(shot_str_list=["I am curious and eager to learn new things, e.g.: I often find myself researching topics that I'm not familiar with, just for the sake of expanding my knowledge.",
                                                         "I enjoy exploring new cultures and ways of life, e.g.: I would rather travel to a country I've never been to before than revisit a familiar destination.",
                                                         "I am a good listener, e.g.: I always try to understand the other person's perspective before responding."], 
                                           num_words=100, 
                                           task_description=task_description)
    message = [
        {"role": "user", "content": prompt}
    ]
    response = LLMFactory.process(message, model_name="GPT-4o", temperature=0.01, max_tokens=1024)
    
    
    print("Prompt: ", prompt)
    print("*" * 50)
    print("Response: ", response)
    reflections = extract_reflection(response)
    print(reflections)

    
            
