import json
from typing import Dict, Tuple, Union
from allennlp.predictors.predictor import Predictor
from allennlp.data.tokenizers.spacy_tokenizer import SpacyTokenizer
from nltk.tokenize import (
    TreebankWordTokenizer,
    word_tokenize,
    ToktokTokenizer,
    PunktSentenceTokenizer,
)
import allennlp_models.structured_prediction
import pandas as pd
from lxml import etree
from tqdm import tqdm
import re
import os
from csv2json14 import get_indices
import utils


def clean_string(string: str):
    string = re.sub(r'[^\w]', ' ', string)
    return string


def str2json(string: str,
             aspect: str,
             outputfile: str):
    tk = SpacyTokenizer()
    print(f'Processing user input: {string}...')
    example = {}
    sent = string.strip()
    term = aspect.split(",")
    sent = " ".join(sent.split(" ")[:-1])
    predictor = Predictor.from_path(
        "https://storage.googleapis.com/allennlp-public-models/biaffine-dependency-parser-ptb-2020.04.06.tar.gz"
    )
    fidx, tidx = get_indices(term, sent)
    fidx_list = []
    tidx_list = []
    fidx_list.append(fidx)
    tidx_list.append(tidx)

    allen = predictor.predict(sentence=sent)
    token, pos, deprel, head, dependencies = utils.dependencies2format(allen)
    example["sentence"] = sent
    example["token"] = list(token)
    example["pos"] = pos
    example["deprel"] = deprel
    example["head"] = head
    example["dependencies"] = dependencies
    example["aspects"] = []

    asp = dict()
    asp["term"] = [str(i) for i in tk.tokenize(term)]
    asp["polarity"] = "neutral"
    left_index = (fidx + 1 + 2 * fidx_list.index(fidx))
    right_index = (tidx + 1 + 2 * tidx_list.index(tidx))
    left_word_offset = len(tk.tokenize(sent[:left_index]))
    to_word_offset = len(tk.tokenize(sent[:right_index]))

    asp["from"] = left_word_offset
    asp["to"] = to_word_offset

    example["aspects"].append(asp)
    examples = [example]
    assert outputfile.endswith('.json')
    with open(outputfile, "w") as f:
        json.dump(examples, f, indent=2)
    print(f'Writing {outputfile}...')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str,
                        default="/home/P76114511/projects/RoBERTaABSA/Dataset/UserInput")
    parser.add_argument("--txtfile", default="input.txt", type=str, help="input file with user input string")
    args = parser.parse_args()
    input_file = os.path.join(args.base_dir, args.txtfile)
    with open(input_file, 'r') as f:
        string = f.readlines()
    string = clean_string(string[0])
    aspect = string[1]
    output_file = os.path.join(args.base_dir, 'Test.json')
    str2json(string, aspect, output_file)
    # clean the string
