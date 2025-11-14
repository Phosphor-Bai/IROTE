# Reflection Optimization

## Code Organization
- data: `reflection_data` stores initialized reflections (use `initialize_demonstrations.ipynb` to generate); `questionnaires.json` should be the questionnaires used for effectiveness evaluation (we do not provide this due to copyright)
- eval_utils: code for evaluation
- models: code for using API
- runs：bash script for IROTE optimization
- optimization.py: main code for IROTE optimization
- prompts.py: prompts used in optimizaion
- probability_estimator.py: code for estimating conditional probabilities (the current approximation of many conditional probabilities is performed by requesting GPT-4o to complete the tasks)
- tools.py: other tools

## Preparation

### Model to download
Download `sup-simcse-roberta-large`(https://huggingface.co/princeton-nlp/sup-simcse-roberta-large) and replace its path to `PATH_TO_SIMCSE_MODEL` in `optimization.py`.

### LLM
We use the API to make requests to the LLM.

If you choose to run IROTE with LLM from OpenAI series (We currently provide implementations for GPT-4o and GPT-4o-mini. Other OpenAI models can also be integrated by simply creating the corresponding class.), you should add environment variable for API query to run the code using `export VARIABLE_NAME=VARIABLE_VALUE`, including:
- GPT_ENDPOINT: Your OpenAI endpoint
- AZURE_OPENAI_API_KEY / OPENAI_API_KEY: Your OpenAI api key, depends on the endpoint you use.

If you choose to run IROTE with  Qwen2.5-7B-Instruct and Mistral-7B-Instruct-v0.3, you should first deploy model using VLLM, e.g. `python3 -m vllm.entrypoints.openai.api_server --model PATH_TO_YOUR_MODEL --served-model-name MODEL_NAME --host 0.0.0.0 --port MODEL_PORT --dtype auto --max-model-len 4096 --tensor-parallel-size 1 --trust-remote-code` (you can adjust the parameters).
Then, change the `QWEN_VLLM_NAME`/`MISTRAL_VLLM_NAME` to MODEL_NAME, and `QWEN_VLLM~PORT`/`MISTRAL_VLLM_PORT` to MODEL_PORT in `models/llm_interface.py`.

If you want to use other LLM, you can modify `models/llm_interface.py` by creating new classes and register in `PRODUCT_MAP` and `chat_models`. If you make API requests through an endpoint, inherit the `CustomOpenAIModel` class.
If the model is deployed via vLLM, directly inherit the `VLLMOpenAIModel` class.

### Initialize Reflections
Run code in `initialize_demonstrations.ipynb` (in the outer folder) to generate initial reflections. This should generate files in `data/reflection_data`.

### Add Questionnaires
We use seven questionnaire in our research, but for IROTE optimization, only four is used: PVQ21 (Schwartz et all, 2001), PVQ-RR (Schwartz 2012), MFQ-1 (Graham et al. 2008), and BFI (John, Donahue, and Kentle 1991). We do not provide the data due to copyright.
Questionnaires should be insert into `data/questionnaire.json` following the structure given in the file.

## Run Optimization
After the preparation, run the following script in `runs` to run the optimization. The optimized file should be created in `output/all_survey`.
If you use other models, you should write its own script.

After optimization, run code in `analyze_res.ipynb` (in the outer folder) to automatically select the best reflections.