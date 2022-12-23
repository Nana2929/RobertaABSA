# exit on error environment
set -e


ptm_type="bert"         # "roberta" or "bert"
datadir="Dataset"
finetuned="ft"          # "ft", "no-ft"
ftdset="Laptop"
dset_name="UserInput"
project_root="/home/P76114511/projects/RoBERTaABSA"
ft_model_path="$project_root/Train/save_models/$ptm_type-$ftdset-FT/$ptm_type"
dset_path="$project_root/$datadir/$dset_name"

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
    echo "(fine-tuned) pretrained model:" $laptop_ptm
    python3 Perturbed-Masking/generate_matrix.py\
        --model_path=$laptop_ptm \
        --data_dir=$datadir \
        --dataset=$dset_name
else
    echo "(no fine-tuned)" $ptm_type
    python3 Perturbed-Masking/generate_matrix.py \
        --model_path=$ptm_type \
        --data_dir=$datadir \
        --dataset=$dset_name
fi
echo == inference ALSC ==
for layer in {0..12}
do
    echo == infering w/ $layer layer ==
    # python3 Perturbed-Masking/generate_asgcn.py --layer=$layer --is_finetuned=$finetuned --matrix_folder="$ptm_type/$dset_name/Train" --root_fp=$project_root
    python3 Perturbed-Masking/generate_asgcn.py --layer=$layer --is_finetuned=$finetuned --matrix_folder="$ptm_type/$dset_name/Test"  --root_fp=$project_root
    python3 ASGCN/infer.py --layer=$layer --input_path=$dset_path/input.txt
    # py infer.py --input_path=/home/P76114511/projects/RoBERTaABSA/Dataset/UserInput/input.txt --layer=3
done;






# generate the alsc result






