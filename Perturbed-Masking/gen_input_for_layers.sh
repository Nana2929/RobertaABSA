
END=$1
for i in $(seq 0 $END);
do
    echo "layer:" $i
    python3 generate_asgcn.py --matrix_folder roberta/Laptop --layers $i
done
