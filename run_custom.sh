#!/bin/bash
datadir="Dataset"
dset="GooglePixel"
textcol="title"         # text title
finetuned=1             # 0: False, 1: True
ptm="roberta"           # if not fine-tuned, use this variable
ftptm="roberta-en"      # if fine-tuned, choose from 'bert-en-base-uncased', 'roberta-en', 'roberta-en-large', 'xlmroberta-xlm-roberta-base', 'bert-multi-base-cased', 'xlnet-xlnet-base-cased'
project_root="/home/P76114511/projects/RoBERTaABSA"
logdirname="logs"
logfile="$project_root/$logdirname/custom_log.txt"
ft_dset="Laptop"
ft_model_path="$project_root/Train/save_models/$ftptm-$ft_dset-FT/$ptm"
# folder = f"{root_fp}/save_models/{model_type}-{args.dataset}-FT"
# run this script at project root:
# TODO: run code
# TODO: add proper log files
# Note: please activate your venv before running. mb activate robertaabsa
python3 ./Dataset/csv2json14.py --base_dir="$datadir/$dset" --textcol=$textcol
if [[ $finetuned -eq 1 ]]
then
    echo == do finetuning ==
    python3 Train/finetune.py --model_name=$ftptm --n_epochs=1 --data_dir=$datadir --dataset=$ft_dset # need to use their dataset
    python3 Perturbed-Masking/generate_matrix.py --model_path=$ft_model_path --data_dir=$datadir --dataset=$dset
else
    echo == no finetuning ==
    python3 Perturbed-Masking/generate_matrix.py --model_path=$ptm --data_dir=$datadir --dataset=$dset
fi
echo == training ALSC ==
for layer in {0..12}
do
    echo == training w/ $layer layer ==
    python3 Perturbed-Masking/generate_asgcn.py --layers=$layer --matrix_folder="$ptm/$dset/Train" --project_dir="./"
    python3 Perturbed-Masking/generate_asgcn.py --layers=$layer --matrix_folder="$ptm/$dset/Test"  --project_dir="./"
    python3 ASGCN/train.py --dataset=asgcn2/$ptm/$layer/$dset --logfile=$logfile --layers=$layer
done;