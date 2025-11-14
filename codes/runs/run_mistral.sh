

model_name='Mistral-7B-Instruct-v0.3'
tasks="survey"  # survey,benchmark
optimization_beam_size=3
summarization_beam_size=2
num_k_shot=3
avg_mode="min" # min, max, avg
top_k=3
words_limit=50
max_iteration=5
init_reflection_num=5
num_item_shuffle=1
use_cot=True
max_tokens=1024
use_task_descriptions=True

for evaluation_system in "value" "personality"
do
	output_dir="output/all_survey_new/${model_name}_${evaluation_system}"

	python optimization.py \
		--output_dir $output_dir \
		--model_name $model_name \
		--evaluation_system $evaluation_system \
		--tasks $tasks \
		--optimization_beam_size $optimization_beam_size \
		--summarization_beam_size $summarization_beam_size \
		--num_k_shot $num_k_shot \
		--avg_mode $avg_mode \
		--top_k $top_k \
		--words_limit $words_limit \
		--max_iteration $max_iteration \
		--init_reflection_num $init_reflection_num \
        --num_item_shuffle $num_item_shuffle \
        --use_cot $use_cot \
        --max_tokens $max_tokens \
        --use_task_description $use_task_descriptions
done
