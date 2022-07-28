eval base_path="~/repos/mt/dictionary_creator"

echo "[INFO] ${base_path}/fast_align/build/fast_align -i ${base_path}/data/$1_$2_$3.txt -d -o -v > ${base_path}/data/forward_$1_$2_$3.align"
${base_path}/fast_align/build/fast_align -i ${base_path}/data/$1_$2_$3.txt -d -o -v > ${base_path}/data/forward_$1_$2_$3.align

echo "[INFO] ${base_path}/fast_align/build/fast_align -i ${base_path}/data/$1_$2_$3.txt -d -o -v -r > ${base_path}/data/reverse_$1_$2_$3.align"
${base_path}/fast_align/build/fast_align -i ${base_path}/data/$1_$2_$3.txt -d -o -v -r > ${base_path}/data/reverse_$1_$2_$3.align

echo "[INFO] ${base_path}/fast_align/build/atools -i ${base_path}/data/forward_$1_$2_$3.align -j ${base_path}/data/reverse_$1_$2_$3.align -c grow-diag-final-and > ${base_path}/data/diag_$1_$2_$3.align"
${base_path}/fast_align/build/atools -i ${base_path}/data/forward_$1_$2_$3.align -j ${base_path}/data/reverse_$1_$2_$3.align -c grow-diag-final-and > ${base_path}/data/diag_$1_$2_$3.align

rm ${base_path}/data/forward_$1_$2_$3.align
rm ${base_path}/data/reverse_$1_$2_$3.align
