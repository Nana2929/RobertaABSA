#! /bin/bash
set -e                     # "exit on error" environment variable
ptm_type="roberta"         # "roberta" or "bert"
finetuned="no-ft"          # "ft", "no-ft"
layer=7                    # 0 - 12, 0 for ptm embedding matrix
while getopts p:f:l: flag
do
    case "${flag}" in
        p) ptm_type=${OPTARG};;
        f) finetuned=${OPTARG};;
        l) layer=${OPTARG};;
    esac
done
echo "* ptm_type: $ptm_type";
echo "* finetuned or not: $finetuned";
echo "* layer: $layer";

project_root="/home/P76114511/projects/RoBERTaABSA"
datadir="Dataset"
ftdset="Laptop"
dset_name="UserInput"
ft_model_path="$project_root/Train/save_models/$ptm_type-$ftdset-FT/$ptm_type"
dset_path="$project_root/$datadir/$dset_name"
state_dict_path="/home/P76114511/projects/RoBERTaABSA/ASGCN/state_dict_finetuned"
# state_dict_path="/home/P76114511/projects/RoBERTaABSA/ASGCN/state_dict"
output_dir="$project_root/UserOutput"
output_path="$output_dir/output.txt"
mkdir -p $output_dir



echo == demo.sh ==
if [[ $ptm_type == "roberta" ]]
then
    laptop_ptm="Train/save_models/roberta-en-Laptop-FT/roberta-en"
elif [[ $ptm_type == "bert" ]]
then
    laptop_ptm="Train/save_models/bert-en-base-uncased-Laptop-FT/bert-en-base-uncased"
fi

python3 ./Dataset/str2json.py --base_dir="$datadir/$dset_name"
if [[ $finetuned == "ft" ]]
then
    echo "(fine-tuned) pretrained model: " $laptop_ptm
    python3 Perturbed-Masking/generate_matrix.py\
        --model_path=$laptop_ptm \
        --data_dir=$datadir \
        --dataset=$dset_name
else
    echo "(no fine-tuned) pretrained model: " $ptm_type
    python3 Perturbed-Masking/generate_matrix.py \
        --model_path=$ptm_type \
        --data_dir=$datadir \
        --dataset=$dset_name
fi

echo == inference ALSC ==
echo == inferring w/ $layer layer ==

python3 Perturbed-Masking/generate_asgcn.py --layer=$layer --is_finetuned=$finetuned --matrix_folder="$ptm_type/$dset_name/Test"  --root_fp=$project_root
python3 ASGCN/infer.py --layer=$layer --input_path=$dset_path/input.txt --state_dict_path=$state_dict_path --alsc_out=$output_path
