'''
Generate model predictions

Input file:
ID1 Sentence1
ID2 Sentence2
.
.
.

Output file:
ID1 Sentence1
ID2 Sentence2
.
.
.
'''

import sys
import os
import argparse
from happytransformer import HappyTextToText, TTSettings
import torch

def get_sentences(data_path):
    with open(data_path, 'r') as f:
        lines = f.readlines()
    texts = [' '.join(l.rstrip('\n').split()[1:]) for l in lines]
    ids = [l.rstrip('\n').split()[0] for l in lines]
    return ids, texts

def correct(model, sentence, gen_args):
    correction_prefix = "grammar: "
    sentence = correction_prefix + sentence
    result = model.generate_text(sentence, gen_args)
    return result.text

if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('IN', type=str, help='Path to input data')
    commandLineParser.add_argument('OUT', type=str, help='Path to corrected output data')
    args = commandLineParser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/predict.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n') 
    
    device = torch.device('cpu')
    # Load Model
    model = HappyTextToText("T5", "vennify/t5-base-grammar-correction")
    model.device = torch.device('cpu')
    gen_args = TTSettings(num_beams=5, min_length=1)

    # Load input sentences
    identifiers, sentences = get_sentences(args.IN)

    # Correction (prediction) for each input sentence
    corrections = []
    for i, sent in enumerate(sentences):
        print(f'On {i}/{len(sentences)}')
        corrections.append(correct(model, sent, gen_args))
    assert len(corrections) == len(identifiers), "Number of ids don't match number of predictions"

    # Save predictions
    with open(args.OUT, 'w') as f:
        for id, sentence in zip(identifiers, corrections):
            f.write(f'{id} {sentence}\n')

