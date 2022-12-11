
"""
Script to convert the Custom Dataset (.csv) to json 14
the format that is required to run the RobertaABSA full code
An example file:
   `RoBERTaABSA/Dataset/GooglePixel/GooglePixel_Train.csv`
Author: Ching Wen Yang
Date: 2022.12.11
Revised based on Dataset/xml2json14.py
"""

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


def get_indices(pattern: str,
                string: str) -> Tuple[int, int]:
    """Return the start and terminal index of the

    Args:
        pattern (str):
        string (str):
    Returns:
        Tuple[int, int]: start (inclusive), terminal (exclusive) index
    """
    for match in re.finditer(pattern, string):
        return match.span()
    return None, None


def csv2txt(file_path, predictor, textcol):
    """
    Read the original xml file of semeval data and extract the text that have aspect terms.
    Store them in txt file.
    file_path: origin_file_path
    """
    # https://colab.research.google.com/drive/1ajhs_-6SuSLW0KzNz-J60WJUN9flGucc?authuser=3#scrollTo=4whEamc8O5bY
    tk = SpacyTokenizer()
    raw = pd.read_csv(file_path)
    print(f'Processing {file_path}...')
    final_out = []
    s_mapping = {'Neg': 'negative', 'Neu': 'neutral', 'Pos': 'positive'}
    for r, row in raw.iterrows():
        example = {}
        sent = row[textcol].strip()
        term = row['aspect_term'].strip()
        pol = row['sentiment_polarity'].strip()
        sent_list = list(sent)  # a list of tokens, not using 'tokens' to avoid confusion with depparser `tokens`

        if pd.isna(term):
            continue  # skip the instance

        # === get indices of the aspect term ===
        # in our Google dataset there's only 1 aspect term in 1 sentence
        fidx, tidx = get_indices(term, sent)

        fidx_list = []
        tidx_list = []
        fidx_list.append(fidx)
        tidx_list.append(tidx)

        if not (fidx or tidx):
            continue
        # add space into sentence before sent
        # Boot time is super fast
        sent_list.insert(fidx, " ")
        sent_list.insert(tidx + 1, " ")
        # /sBoot time/s  is super fast # add /s
        sent = "".join(sent_list)
        example['sentence'] = sent

        # === dependency parsing ===
        allen = predictor.predict(sentence=sent)
        token, pos, deprel, head, dependencies = dependencies2format(allen)
        example["token"] = list(token)
        example["pos"] = pos
        example["deprel"] = deprel
        example["head"] = head
        example["dependencies"] = dependencies
        example["aspects"] = []

        # === dealing with aspect term(s): asp ===
        norm_pol = s_mapping.get(pol, 'neutral')

        asp = dict()
        asp["term"] = [str(i) for i in tk.tokenize(term)]
        asp["polarity"] = norm_pol
        left_index = (fidx + 1 + 2 * fidx_list.index(fidx))
        right_index = (tidx + 1 + 2 * tidx_list.index(tidx))
        left_word_offset = len(tk.tokenize(sent[:left_index]))
        to_word_offset = len(tk.tokenize(sent[:right_index]))

        asp["from"] = left_word_offset
        asp["to"] = to_word_offset

        example["aspects"].append(asp)
        final_out.append(example)
        extended_filename = file_path.replace('.csv', '.json')
        with open(extended_filename, "w") as f:
            json.dump(final_out, f, indent=2)
    print(f'Writing {extended_filename}...')
    print(f"Done with {len(final_out)} data entries.")


def dependencies2format(doc):  # doc.sentences[i]
    """
    Format annotation: sentence of keys
                                - tokens
                                - tags
                                - predicted_dependencies
                                - predicted_heads
                                - dependencies
    RETURN token,pos,deprel,head,dependencies
    """
    token = doc["words"]
    pos = doc["pos"]
    # sentence['energy'] = doc['energy']
    predicted_dependencies = doc["predicted_dependencies"]
    predicted_heads = doc["predicted_heads"]
    deprel = doc["predicted_dependencies"]
    head = doc["predicted_heads"]
    dependencies = []
    for idx, item in enumerate(predicted_dependencies):
        dep_tag = item
        frm = predicted_heads[idx]
        to = idx + 1
        dependencies.append(
            [dep_tag, frm, to]
        )

    return token, pos, deprel, head, dependencies


def get_all_file(args):
    import os
    base_dir = args.base_dir
    textcol = args.textcol
    csv2txt(os.path.join(base_dir, "Test.csv"), predictor, textcol)
    csv2txt(os.path.join(base_dir, "Train.csv"), predictor, textcol)


predictor = Predictor.from_path(
    "https://storage.googleapis.com/allennlp-public-models/biaffine-dependency-parser-ptb-2020.04.06.tar.gz"
)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str,
                        default="/home/P76114511/projects/RoBERTaABSA/Dataset/GooglePixel")
    parser.add_argument("--textcol", type=str, default="title")
    args = parser.parse_args()
    get_all_file(args)
