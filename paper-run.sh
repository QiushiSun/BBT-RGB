#!/bin/bash

startTime=`date +%Y%m%d-%H:%M`
startTime_s=`date +%s`

seed_lst=(42)
task_name_lst=(RTE) # SST-2 Yelp AGNews DBPedia SNLI MRPC RTE
offset_lst=(1000)
device=cuda:0
model_name=roberta-large
model_path=transformer_model/roberta-large
loss_type=ce 
sigma1=1
sigma2=0.2
budget=8000
budget2=0
data_dir=paper-datasets-aligned
eval_every=100
alpha=0.5
# --instruction
# --in_contexts 
# --multiVerbalizer

for task_name in "${task_name_lst[@]}"; do
    for seed in "${seed_lst[@]}"; do
        for offset in "${offset_lst[@]}"; do
            python -u deepbbt-for-paper.py --budget2 $budget2 --alpha $alpha --data_dir $data_dir --seed $seed --task_name $task_name --device $device --budget $budget --loss_type $loss_type --sigma1 $sigma1 --sigma2 $sigma2 --model_name $model_name --model_path $model_path --eval_every $eval_every --offset $offset
        done
    done
done

# task_name=SST-2  
# device=cuda:0
# results_dir_lst=(SST-2_Ins_offset1000_Cobyla6000/last_epoch_results)

# for results_dir in "${results_dir_lst[@]}"; do
#     python -u test_deepbbt.py --offline --instruction --task_name $task_name --device $device --results_dir $results_dir
# done

endTime=`date +%Y%m%d-%H:%M`
endTime_s=`date +%s`
sumTime=$[ $endTime_s - $startTime_s ]
echo "$startTime ---> $endTime" "Totl:$sumTime seconds" 