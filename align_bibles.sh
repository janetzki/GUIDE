# usage: align_bibles.sh <input1_path> <input2_path> <suffix> <output_path>

eval input1_path="$1"
eval input2_path="$2"
eval suffix="$3"
eval base_path="~/repos/mt/dictionary_creator/"
eval file_path="$(pwd)/$4/${input1_path}_${input2_path}_${suffix}"

"${base_path}"/fast_align/build/fast_align -i "${file_path}.txt" -d -o -v > "${file_path}_forward.align"
"${base_path}"/fast_align/build/fast_align -i "${file_path}.txt" -d -o -v -r > "${file_path}_reverse.align"
"${base_path}"/fast_align/build/atools -i "${file_path}_forward.align" -j "${file_path}_reverse.align" -c grow-diag-final-and > "${file_path}_diag.align"

rm "${file_path}_forward.align"
rm "${file_path}_reverse.align"
