#!/bin/sh -e

# Create the folder structure
mkdir data || true
mkdir data/0_state || true
mkdir data/1_aligned_bibles || true
mkdir data/2_sd_labeling || true
mkdir data/3_models || true
mkdir data/4_semdoms || true
mkdir data/5_raw_bibles || true
mkdir data/5_raw_bibles/eBible || true
mkdir "data/5_raw_bibles/eBible/no semdoms available" || true
mkdir data/6_results || true
mkdir data/7_graphs || true
mkdir data/8_plots || true
mkdir data/9_datasets || true
mkdir data/10_lemmatized_bibles || true

# Download git LFS files
git lfs install
git lfs pull

# Download the raw bibles
curl https://raw.githubusercontent.com/BibleNLP/ebible/main/corpus/ben-ben2017.txt >data/5_raw_bibles/eBible/ben-ben2017.txt
curl https://raw.githubusercontent.com/BibleNLP/ebible/main/corpus/cmn-cmn-cu89s.txt >data/5_raw_bibles/eBible/cmn-cmn-cu89s.txt
curl https://raw.githubusercontent.com/BibleNLP/ebible/main/corpus/eng-eng-web.txt >data/5_raw_bibles/eBible/eng-eng-web.txt
curl https://raw.githubusercontent.com/BibleNLP/ebible/main/corpus/fra-frasbl.txt >data/5_raw_bibles/eBible/fra-frasbl.txt
curl https://raw.githubusercontent.com/BibleNLP/ebible/main/corpus/hin-hin2017.txt >data/5_raw_bibles/eBible/hin-hin2017.txt
curl https://raw.githubusercontent.com/BibleNLP/ebible/main/corpus/ind-ind.txt >data/5_raw_bibles/eBible/ind-ind.txt
curl https://raw.githubusercontent.com/BibleNLP/ebible/main/corpus/mkn-mkn.txt >data/5_raw_bibles/eBible/mkn-mkn.txt
curl https://raw.githubusercontent.com/BibleNLP/ebible/main/corpus/mal-mal.txt >data/5_raw_bibles/eBible/mal-mal.txt
curl https://raw.githubusercontent.com/BibleNLP/ebible/main/corpus/npi-npiulb.txt >data/5_raw_bibles/eBible/npi-npiulb.txt
curl https://raw.githubusercontent.com/BibleNLP/ebible/main/corpus/por-porbrbsl.txt >data/5_raw_bibles/eBible/por-porbrbsl.txt
curl https://raw.githubusercontent.com/BibleNLP/ebible/main/corpus/spa-spablm.txt >data/5_raw_bibles/eBible/spa-spablm.txt
curl https://raw.githubusercontent.com/BibleNLP/ebible/main/corpus/swh-swhulb.txt >data/5_raw_bibles/eBible/swh-swhulb.txt
curl https://raw.githubusercontent.com/BibleNLP/ebible/main/corpus/deu-deu1951.txt >"data/5_raw_bibles/eBible/no semdoms available/deu-deu1951.txt"
curl https://raw.githubusercontent.com/BibleNLP/ebible/main/corpus/hmo-hmo.txt >"data/5_raw_bibles/eBible/no semdoms available/hmo-hmo.txt"
curl https://raw.githubusercontent.com/BibleNLP/ebible/main/corpus/meu-meu.txt >"data/5_raw_bibles/eBible/no semdoms available/meu-meu.txt"
curl https://raw.githubusercontent.com/BibleNLP/ebible/main/corpus/azb-azb.txt >"data/5_raw_bibles/eBible/no semdoms available/azb-azb.txt"
curl https://raw.githubusercontent.com/BibleNLP/ebible/main/corpus/tpi-tpi.txt >"data/5_raw_bibles/eBible/no semdoms available/tpi-tpi.txt"
curl https://raw.githubusercontent.com/BibleNLP/ebible/main/corpus/yor-yor.txt >"data/5_raw_bibles/eBible/no semdoms available/yor-yor.txt"

# Create the conda environment
conda create --name guide_env python=3.11
eval "$(conda shell.bash hook)"
conda activate guide_env
y | conda env update -n guide_env -f environment.yml
yes | pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cpu.html

# Add the submodules
git submodule add https://github.com/clab/fast_align.git fast_align/
git submodule add https://github.com/robertostling/eflomal.git eflomal/
git submodule update --init --recursive

# Install fast_align
mkdir fast_align/build || true
cd fast_align/build || exit
cmake ..
make
cd ../..

# Install Eflomal
yes | pip install eflomal
yes | python -m pip install eflomal/
