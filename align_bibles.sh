# usage: align_bibles.sh <file_path>

# eval base_path="~/repos/mt/dictionary_creator/"
eval file_path="$(pwd)/$1"

# "${base_path}"/fast_align/build/fast_align -i "${file_path}.txt" -d -o -v > "${file_path}_forward.align"
# "${base_path}"/fast_align/build/fast_align -i "${file_path}.txt" -d -o -v -r > "${file_path}_reverse.align"
python eflomal/build/scripts-3.11/eflomal-align -i "${file_path}.txt" --model 3 -f "${file_path}_eflomal_forward.align" -r "${file_path}_eflomal_reverse.align"
# "${base_path}"/fast_align/build/atools -i "${file_path}_forward.align" -j "${file_path}_reverse.align" -c grow-diag-final-and > "${file_path}_diag.align"
fast_align/build/atools -i "${file_path}_eflomal_forward.align" -j "${file_path}_eflomal_reverse.align" -c grow-diag-final-and > "${file_path}_eflomal_diag.align"

# rm "${file_path}_forward.align"
# rm "${file_path}_reverse.align"
rm "${file_path}_eflomal_forward.align"
rm "${file_path}_eflomal_reverse.align"