#!/bin/bash

startTime=`date +%Y%m%d-%H:%M`
startTime_s=`date +%s`

# seed_lst=(8 13 42 50 60) # 8 13 42 50 60
task_name_lst=(SNLI RTE DBPedia AGNews MRPC Yelp SST-2)  # SST-2 Yelp AGNews TREC MRPC SNLI
# task_name_lst=(RTE)
device=cuda:3
cuda=3
model_name=roberta-large
model_path=transformer_model/roberta-large
n_prompt_tokens=50
intrinsic_dim=500
k_shot=16
loss_type=ce  # hinge ce
random_proj=normal
sigma1=1
sigma2=0.2
popsize=20
bound=0
budget=8000
print_every=50
eval_every=1
data_dir=paper-datasets-aligned  # datasets_augment_pos_final  # datasets_augment_pos_final
# --use_rlprompt --multiVerbalizer

for task_name in "${task_name_lst[@]}"; do
    if [[ $task_name == SST-2 || $task_name == SNLI ]]; then 
        # seed_lst=(8 13 42 50 60)
        seed_lst=(42)
    else
        seed_lst=(42)
        # seed_lst=(8 42 50 100 717)
    fi

    for seed in "${seed_lst[@]}"; do
        python -u incontext_select.py --instruction --in_contexts --data_dir $data_dir --seed $seed --task_name $task_name --device $device --budget $budget --model_name $model_name --model_path $model_path --eval_every $eval_every
    done
    # python -u test.py --task_name $task_name --cuda $cuda
done

python -u incontext_select_final.py




endTime=`date +%Y%m%d-%H:%M`
endTime_s=`date +%s`
sumTime=$[ $endTime_s - $startTime_s ]
echo "$startTime ---> $endTime" "Totl:$sumTime seconds" 