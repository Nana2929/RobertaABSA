END=$1
for i in $(seq 0 $END);
do
    echo "layer:" $i
    python3 train.py --dataset /home/P76114511/RoBERTaABSA/Perturbed-Masking/asgcn2/roberta/$i/Laptop \
	--save True > training_logs/log-$i
done