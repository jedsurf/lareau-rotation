import sys
import os
sys.path.append("src/")
from src import *
sys.path.append("scripts/")
from ixnos import encode, iXnos
import torch
from torch import nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from Bio import SeqIO
import seaborn as sns
import random

ansari_n5p4 = iXnos(n_codons=10)
ansari_n5p4.load_state_dict(torch.load("processed-data/thp1_v2/models/ixnos_n5p4_full.pth"))
ansari_n5p4.eval() 

ansari_n3p2 = iXnos(n_codons=6)
ansari_n3p2.load_state_dict(torch.load("processed-data/thp1_v2/models/ixnos_n3p2_full.pth"))
ansari_n3p2.eval() 

iwasaki_n5p4 = iXnos(n_codons=10)
iwasaki_n5p4.load_state_dict(torch.load("models/ixnos_retrained.pth"))
iwasaki_n5p4.eval() 

iwasaki_n3p2 = iXnos(n_codons=6)
iwasaki_n3p2.load_state_dict(torch.load("processed-data/iwasaki/models/ixnos_n3p2_full.pth"))
iwasaki_n3p2.eval() 

cit_seq = 'MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLGYGLMCFARYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSYQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK'

let2cod = {
    'F': ['TTT', 'TTC'],
    'L': ['TTA', 'TTG', 'CTT', 'CTC', 'CTA', 'CTG'],
    'I': ['ATT', 'ATC', 'ATA'],
    'M': ['ATG'],
    'V': ['GTT', 'GTC', 'GTA', 'GTG'],
    'S': ['TCT', 'TCC', 'TCA', 'TCG', 'AGT', 'AGC'],
    'P': ['CCT', 'CCC', 'CCA', 'CCG'],
    'T': ['ACT', 'ACC', 'ACA', 'ACG'],
    'A': ['GCT', 'GCC', 'GCA', 'GCG'],
    'Y': ['TAT', 'TAC'],
    'H': ['CAT', 'CAC'],
    'Q': ['CAA', 'CAG'],
    'N': ['AAT', 'AAC'],
    'K': ['AAA', 'AAG'],
    'D': ['GAT', 'GAC'],
    'E': ['GAA', 'GAG'],
    'C': ['TGT', 'TGC'],
    'W': ['TGG'],
    'R': ['CGT', 'CGC', 'CGA', 'CGG', 'AGA', 'AGG'],
    'G': ['GGT', 'GGC', 'GGA', 'GGG'],
}

def get_inputs(codons : list):
    # Given a list of codons, get an input vector for iXnos
    nts = "".join(codons)
    codon_vector = np.concatenate([encode(i, iXnos.get_codon_to_id()) for i in codons])
    nt_vector = np.concatenate([encode(i, iXnos.get_nt_to_id()) for i in nts])
    input_vector = np.concatenate([codon_vector, nt_vector])
    input_vector = torch.from_numpy(input_vector).to(torch.float32)
    return input_vector

def predict_elongation(seq, model, min_cod=-5, max_cod=4):
    """Given a sequence and an iXnos model, predict the sum of
    scaled counts at each codon index.

    Args:
        seq (str): Amino acid sequence of transcript of interest.
        model (iXnos model): iXnos model to use
        min_cod (int, optional): Minimum index of codons used in iXnos model. Defaults to -5.
        max_cod (int, optional): Maximum index of codons used in iXnos model. Defaults to 4.

    Returns:
        int: Sum of scaled counts; proxy for predicted elongation time
    """    
    seq = seq.upper()
    # I train iXnos on DNA sequences, so need to convert U to T
    if "U" in seq:
        seq = seq.replace("U", "T")
    # Add "NNN" codons to beginning and end of sequence in order to pass 
    # first and last few codons through iXnos
    seq = "".join(["NNN" for i in range(0 - min_cod)]) \
        + seq + "".join(["NNN" for i in range(max_cod)])
    # Predict scaled counts across all codons
    codons = [seq[i:i+3] for i in range(0, len(seq), 3)]
    overall_count = 0
    for i in range(0 - min_cod, len(codons) - max_cod):
        input_vector = get_inputs(codons[i + min_cod:i + max_cod + 1])
        overall_count += model(input_vector).item()
    return overall_count

def predict_random_speeds(cds, n_samples, model, **kwargs):
    seqs, speeds = [], []
    for i in range(n_samples):
        codons = [random.choice(let2cod[i]) for i in cds]
        nt_seq = "".join(codons)
        pred_speed = predict_elongation(nt_seq, model, **kwargs)
        seqs.append(nt_seq)
        speeds.append(pred_speed)
    return seqs, speeds

n = 100_000
mode = int(sys.argv[1])
if mode == 1:
    seqs, speeds_iwasaki = predict_random_speeds(cit_seq, n, iwasaki_n3p2, min_cod=-3, max_cod=2)
    speeds_ansari = [predict_elongation(i, ansari_n3p2, -3, 2) for i in seqs]  
    result = pd.DataFrame({
        "seq" : seqs,
        "speeds_iwasaki" : speeds_iwasaki,
        "speeds_ansari" : speeds_ansari,
    })
    result.to_csv("processed-data/iwasaki_vs_ansari_ecitrine_n3p2.csv", index=False)
elif mode == 2:
    seqs, speeds_iwasaki = predict_random_speeds(cit_seq, n, iwasaki_n5p4, min_cod=-5, max_cod=4)
    speeds_ansari = [predict_elongation(i, ansari_n5p4, -5, 4) for i in seqs]  
    result = pd.DataFrame({
        "seq" : seqs,
        "speeds_iwasaki" : speeds_iwasaki,
        "speeds_ansari" : speeds_ansari,
    })
    result.to_csv("processed-data/iwasaki_vs_ansari_ecitrine_n5p4.csv", index=False)
else:
    print("Must specify mode!\n\t1: -3 to +2 model\n\t2: -5 to +4 model")