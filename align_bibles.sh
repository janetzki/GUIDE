# usage: align_bibles.sh <input1_path> <input2_path> <suffix> <output_path>

eval input1_path="$1"
eval input2_path="$2"
eval suffix="$3"
eval base_path="~/repos/mt/dictionary_creator/"
eval file_path="$(pwd)/$4/${input1_path}_${input2_path}_${suffix}"

# # fast_align
# "${base_path}"/fast_align/build/fast_align -i "${file_path}.txt" -d -o -v > "${file_path}_forward.align"
# "${base_path}"/fast_align/build/fast_align -i "${file_path}.txt" -d -o -v -r > "${file_path}_reverse.align"
# "${base_path}"/fast_align/build/atools -i "${file_path}_forward.align" -j "${file_path}_reverse.align" -c grow-diag-final-and > "${file_path}_diag.align"
#
# rm "${file_path}_forward.align"
# rm "${file_path}_reverse.align"

DATA_FILE="${file_path}.txt"
MODEL_NAME_OR_PATH=bert-base-multilingual-cased
OUTPUT_FILE="${file_path}_awesome.align"

# code from https://github.com/neulab/awesome-align/blob/master/README.md
# fine-tuning
#CUDA_VISIBLE_DEVICES=0 awesome-train \
#    --output_dir=$OUTPUT_DIR \
#    --model_name_or_path=bert-base-multilingual-cased \
#    --extraction 'softmax' \
#    --do_train \
#    --train_tlm \
#    --train_so \
#    --train_data_file=$DATA_FILE \
#    --per_gpu_train_batch_size 2 \
#    --gradient_accumulation_steps 4 \
#    --num_train_epochs 1 \
#    --learning_rate 2e-5 \
#    --save_steps 4000 \
#    --max_steps 20000 #\
#    #--do_eval \
#    #--eval_data_file=$EVAL_FILE

CUDA_VISIBLE_DEVICES=0 awesome-align \
    --output_file=$OUTPUT_FILE \
    --model_name_or_path=$MODEL_NAME_OR_PATH \
    --data_file=$DATA_FILE \
    --extraction 'softmax' \
    --batch_size 32

