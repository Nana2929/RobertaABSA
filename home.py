import streamlit as st
import base64
import os
from tree_drawer.drawer import DepTree

st.title("Deep Learning Final Project Demo")

with st.form("form", clear_on_submit=True):
    text = st.text_input("Sentence")
    aspect = st.text_input("Aspect term")
    model = st.radio("Model", ["Roberta", "BERT"], horizontal=True)
    ft = st.radio("Fine-tuned?", ["Yes", "No"], horizontal=True)
    layer = st.slider("layer", 1, 12)
    inference = st.form_submit_button("Inference")

### render svg file
def render_svg(svg):
    """Renders the given svg string."""
    b64 = base64.b64encode(svg.encode('utf-8')).decode("utf-8")
    html = r'<img src="data:image/svg+xml;base64,%s"/>' % b64
    st.write(html, unsafe_allow_html=True)

### variables and paths
input_dir = "/data/fangyi/Final/RobertaABSA/Dataset/UserInput"

if model == "Roberta":
    ptm_type = "roberta"         # "roberta" or "bert"
    laptop_ptm = "Train/save_models/roberta-en-Laptop-FT/roberta-en"
else:
    ptm_type = "bert"
    laptop_ptm = "Train/save_models/bert-en-base-uncased-Laptop-FT/bert-en-base-uncased"
datadir = "Dataset"
if ft == "Yes":
    finetuned = "ft"          # "ft", "no-ft"
else:
    finetuned = "no-ft"
ftdset = "Laptop"
dset_name = "UserInput"
project_root = "/data/fangyi/Final/RobertaABSA"
ft_model_path = f"{project_root}/Train/save_models/$ptm_type-{ftdset}-FT/{ptm_type}"
dset_path = f"{project_root}/{datadir}/{dset_name}"
state_dict_path = f"{project_root}/ASGCN/state_dict_finetuned"
output_dir = f"{project_root}/UserOutput"
output_path = f"{output_dir}/output.txt"
json_path = f"{dset_path }/Test.json"
npy_path = f"/data/fangyi/Final/RobertaABSA/DepTrees/UserInput-Test-{layer-1}.npy"


if inference:
    if text == "" or aspect == "":
        st.error('Please enter your sentence and aspect term.', icon="❗")
    else:
        if aspect not in text:
            st.error('Please make sure that the specified aspect term is in the input sentence.', icon="🚨")
        else:
            st.write(f"Sentence: {text}")
            st.write(f"Aspect term: {aspect}")
            with open("/data/fangyi/Final/RobertaABSA/Dataset/UserInput/input.txt", "w") as file:
                file.write(f"{text}\n")
                file.write(f"{aspect}") 
            with st.spinner("Model processing..."):
                os.system(f'python3 ./Dataset/str2json.py --base_dir="{datadir}/{dset_name}"')
                if ft == "Yes":
                    os.system(f'python3 Perturbed-Masking/generate_matrix.py \
                                --model_path={laptop_ptm} \
                                --data_dir={datadir} \
                                --dataset={dset_name}')
                else:
                    os.system(f'python3 Perturbed-Masking/generate_matrix.py \
                                --model_path={ptm_type} \
                                --data_dir={datadir} \
                                --dataset={dset_name}')
                os.system(f'python3 Perturbed-Masking/generate_asgcn.py \
                            --layer={layer-1} \
                            --is_finetuned={finetuned} \
                            --matrix_folder="{ptm_type}/{dset_name}/Test"  \
                            --root_fp={project_root}')
                os.system(f'python3 ASGCN/infer.py \
                            --layer={layer-1} \
                            --input_path={dset_path}/input.txt \
                            --state_dict_path={state_dict_path}   ')
                with open(output_path, "r") as f:
                    lines = f.readlines()
                st.write(lines[0].strip("\n"))
                st.write(lines[1])
                # st.write("done!")
            TreeDrawer = DepTree(jsonfile=json_path,npyfile=npy_path)
            svg = TreeDrawer.draw()
            with st.container():
                 render_svg(svg)
