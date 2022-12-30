#!/bin/bash
set -e # exit on error
datadir="Dataset"
dset="GooglePixel"
textcol="title"                       # text title

finetuned="ft"                       # "ft", "no-ft"
ftptm="roberta-en"        # if fine-tuned, choose from 'bert-en-base-uncased', 'roberta-en', 'roberta-en-large', 'xlmroberta-xlm-roberta-base', 'bert-multi-base-cased', 'xlnet-xlnet-base-cased'
ptm="roberta"                          # otherwise, choose from 'bert', 'roberta', 'xlmroberta', 'xlnet'

project_root="/home/P76114511/projects/RoBERTaABSA"
ftdset="Laptop"
ft_model_path="$project_root/Train/save_models/$ftptm-$ftdset-FT/$ftptm"

# run this script at project root:
# Note: please activate your venv before running. mb activate robertaabsa
python3 ./Dataset/csv2json14.py --base_dir="$datadir/$dset" --textcol=$textcol
if [[ $finetuned == "ft" ]]
then
    echo == do finetuning ==
    python3 Train/finetune.py --root_fp=$project_root\
    --model_name=$ftptm --n_epochs=5 --data_dir=$datadir\
    --dset_name=$dset\
    --is_finetuned=$finetuned\
    --dataset=$ftdset
    python3 Perturbed-Masking/generate_matrix.py --model_path=$ft_model_path --data_dir=$datadir --dataset=$dset
else
    echo == no finetuning ==
    python3 Perturbed-Masking/generate_matrix.py --model_path=$ptm --data_dir=$datadir --dataset=$dset
fi

echo == training ALSC ==
for layer in {0..12}
do
    echo == training w/ $layer layer ==
    python3 Perturbed-Masking/generate_asgcn.py --layer=$layer --is_finetuned=$finetuned --matrix_folder="$ptm/${dset}_${finetuned}/Train" --root_fp=$project_root
    python3 Perturbed-Masking/generate_asgcn.py --layer=$layer --is_finetuned=$finetuned --matrix_folder="$ptm/${dset}_${finetuned}/Test"  --root_fp=$project_root
    python3 ASGCN/train.py --dataset=asgcn2/$ptm/$layer/${dset}_${finetuned} \
    --is_finetuned=$finetuned --ptm_name=$ptm --root_fp=$project_root --layers=$layer
done;