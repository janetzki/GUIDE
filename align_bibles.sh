eval base_path="~/repos/mt/experiments"

echo "[INFO] ${base_path}/fast_align/build/fast_align -i ${base_path}/data/$1-$2.txt -d -o -v > ${base_path}/data/forward-$1-$2.align"
${base_path}/fast_align/build/fast_align -i ${base_path}/data/$1-$2.txt -d -o -v > ${base_path}/data/forward-$1-$2.align

echo "[INFO] ${base_path}/fast_align/build/fast_align -i ${base_path}/data/$1-$2.txt -d -o -v -r > ${base_path}/data/reverse-$1-$2.align"
${base_path}/fast_align/build/fast_align -i ${base_path}/data/$1-$2.txt -d -o -v -r > ${base_path}/data/reverse-$1-$2.align

echo "[INFO] ${base_path}/fast_align/build/atools -i ${base_path}/data/forward-$1-$2.align -j ${base_path}/data/reverse-$1-$2.align -c grow-diag-final-and > ${base_path}/data/diag-$1-$2.align"
${base_path}/fast_align/build/atools -i ${base_path}/data/forward-$1-$2.align -j ${base_path}/data/reverse-$1-$2.align -c grow-diag-final-and > ${base_path}/data/diag-$1-$2.align
