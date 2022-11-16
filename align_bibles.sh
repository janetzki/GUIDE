eval base_path="~/repos/mt/dictionary_creator"
eval output_path=$4

"${base_path}"/fast_align/build/fast_align -i "${base_path}/${output_path}/$1_$2_$3.txt" -d -o -v > "${base_path}/${output_path}/forward_$1_$2_$3.align"
"${base_path}"/fast_align/build/fast_align -i "${base_path}/${output_path}/$1_$2_$3.txt" -d -o -v -r > "${base_path}/${output_path}/reverse_$1_$2_$3.align"
"${base_path}"/fast_align/build/atools -i "${base_path}/${output_path}/forward_$1_$2_$3.align" -j "${base_path}/${output_path}/reverse_$1_$2_$3.align" -c grow-diag-final-and > "${base_path}/${output_path}/diag_$1_$2_$3.align"

rm "${base_path}/${output_path}/forward_$1_$2_$3.align"
rm "${base_path}/${output_path}/reverse_$1_$2_$3.align"
