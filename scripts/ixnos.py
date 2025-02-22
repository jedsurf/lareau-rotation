import torch
from torch import nn
import sys
import os
import argparse
from collections import OrderedDict
import numpy as np
from Bio import SeqIO
import pandas as pd
from scipy.stats import pearsonr
from torch.utils.data import DataLoader, TensorDataset
import glob
import random
import itertools
import pickle

# GENERAL SCRIPT to run iXnos on ribo-seq data
# Sample linear leaveout series command:
    # for i in {-5..4}; do python ixnos.py -d ../processed-data/thp1 -o ../processed-data/thp1/models -g ../iXnos/genome_data/human.transcripts.13cds10.transcripts.fa -l $i > ../processed-data/thp1/logs/leaveout_$i.log 2>&1; done

class iXnos(nn.Module):
    def __init__(self, min_codon = -5, max_codon = 4, leaveout:int=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Get general useful properties
        self.codon_to_id = self.get_codon_to_id()
        self.nt_to_id = self.get_nt_to_id()
        self.let2cod = self.get_aa_to_codon()
        # iXnos-specific stuff
        self.min_codon = min_codon
        self.max_codon = max_codon
        self.codon_indices = list(range(self.min_codon, self.max_codon + 1))
        self.nt_indices = list(range(3 * self.min_codon, 3 * (self.max_codon + 1)))
        self.leaveout = leaveout
        if self.leaveout is not None:
            self.codon_indices.remove(self.leaveout)
            for i in range(3 * self.leaveout, 3 * (self.leaveout + 1)):
                self.nt_indices.remove(i)
        self.n_codons = len(self.codon_indices)
        self.input_layer_size = 76 * self.n_codons
        self.layers = nn.Sequential(
            nn.Linear(self.input_layer_size, 200),
            nn.Tanh(),
            nn.Linear(200, 1),
            nn.ReLU(),
        )
    
    def forward(self, x):
        output = self.layers(x)
        return output 


    def predict(self, x):
        """Predicts scaled codon count at a given codon position"""
        self.eval()
        with torch.no_grad():
            return self(x)
    
    def get_inputs(self, codons : list):
        # Given a list of codons, get an input vector for iXnos
        nts = "".join(codons)
        codon_vector = np.concatenate([self.encode(i, self.codon_to_id) for i in codons])
        nt_vector = np.concatenate([self.encode(i, self.nt_to_id) for i in nts])
        input_vector = np.concatenate([codon_vector, nt_vector])
        input_vector = torch.from_numpy(input_vector).to(torch.float32)
        return input_vector
    
    def predict_elongation(self, seq, profile=False):
        """Given a sequence and an iXnos model, predict the sum of
        scaled counts at each codon index.

        Args:
            seq (str): Amino acid sequence of transcript of interest.
            profile (bool): If true, returns a list of the scaled counts at each position. 
                            Defaults to False. 

        Returns:
            int: Sum of scaled counts; proxy for predicted elongation time
        """    
        seq = seq.upper()
        # I train iXnos on DNA sequences, so need to convert U to T
        if "U" in seq:
            seq = seq.replace("U", "T")
        # Add "NNN" codons to beginning and end of sequence in order to pass 
        # first and last few codons through iXnos
        seq = "".join(["NNN" for i in range(0 - self.min_codon)]) \
            + seq + "".join(["NNN" for i in range(self.max_codon)])
        # Predict scaled counts across all codons
        codons = [seq[i:i+3] for i in range(0, len(seq), 3)]
        if profile:
            counts = []
            for i in range(0 - self.min_codon, len(codons) - self.max_codon):
                input_vector = self.get_inputs(codons[i + self.min_codon:i + self.max_codon + 1])
                counts.append(self.predict(input_vector).item())
            return counts
        else:
            overall_count = 0
            for i in range(0 - self.min_codon, len(codons) - self.max_codon):
                input_vector = self.get_inputs(codons[i + self.min_codon:i + self.max_codon + 1])
                overall_count += self.predict(input_vector).item()
            return overall_count

    def predict_random_speeds(self, cds, n_samples, **kwargs):
        seqs, speeds = [], []
        for i in range(n_samples):
            codons = [random.choice(self.let2cod[i]) for i in cds]
            nt_seq = "".join(codons)
            pred_speed = self.predict_elongation(nt_seq, **kwargs)
            seqs.append(nt_seq)
            speeds.append(pred_speed)
        return seqs, speeds
    
    def optimize_codons(self, seq: str, fastest = True, stop_codon = True, sanity_check = False):
        """Computes the coding sequence for a given amino acid sequence that minimizes or maximizes 
        overall counts predicted by this model.

        Args:
            seq (str): String amino acid sequence to optimize. 
            fastest (bool, optional): Specify true to find the CDS that minimizes overall counts 
                or False to find the CDS that maximizes overall counts. Defaults to True.
            stop_codon (bool, optional): Whether or not to include a stop codon at the end of the 
                optimized sequence. Defaults to True. 

        Returns:
            str: Optimal coding sequence as a string
        """        
        let2cod = iXnos.get_aa_to_codon()
        seq = seq.upper()
        #TODO: ADD * TO END
        if stop_codon:
            if seq[-1] != "*":
                seq = seq + "*"
        def zeta(aa_seq):
            # Generate a list of all possible codon combinations for a given AA sequence
            possible_codons = [let2cod[i] for i in aa_seq]
            if len(aa_seq) < self.max_codon + 1 - self.min_codon:
                possible_codons = [["NNN"] for _ in range(self.max_codon + 1 - self.min_codon - len(aa_seq))] + possible_codons
            return list(itertools.product(*possible_codons))
        
        
        L = len(seq)
        # Q[i] contains possible codon combinations within the iXnos window at position i
        # T[i] contains the iXnos predictions for each combination in Q[i]
        Q, T = [], [] 
        c_min = np.zeros(L, dtype=int)
        c_max = np.zeros(L, dtype=int)
        # Iterate through the given AA sequence, get combos of possible codon combinations at 
        # each pertinent iXnos window, and calculate + store iXnos prediction for each codon set
        for i in range(L):
            c_min[i] = max(0, i + self.min_codon - self.max_codon)
            c_max[i] = i + 1
            Q.append(zeta(seq[c_min[i]: c_max[i]]))
            # Calculate expected ribo counts
            T.append([])
            for q in Q[i]:
                prediction_q = self.predict(self.get_inputs(q)).item()
                T[i].append(prediction_q)
        # Calculate optimal preceding sequence sets for each sequence set
        # P[i] stores the index of the shortest path so far running through each possible sequence combo in Q
        # V[i] stores the running sum of iXnos-predicted counts at each position i
        P, V = [], []
        for i in range(L):
            P.append([])
            V.append([])
            if i == 0:
                for q_i, q in enumerate(Q[i]):
                    P[i].append(np.nan)
                    V[i].append(T[i][q_i])
            else:
                for q_i, q in enumerate(Q[i]):
                    # Get indices of previous aa that match the possible codon sequences preceding this aa
                    previous_indices = np.where([q_prev[1:] == q[:-1] for q_prev in Q[i - 1]])[0]
                    # Find either the fastest or slowest possible path that ends in this particular codon combo
                    # TODO: Potentially alter this to consider multiple possible paths
                    if fastest:
                        P_i_q = int(previous_indices[np.argmin([V[i - 1][j] for j in previous_indices])])
                    else:
                        P_i_q = int(previous_indices[np.argmax([V[i - 1][j] for j in previous_indices])])
                    P[i].append(P_i_q)
                    V[i].append(V[i-1][P_i_q] + T[i][q_i])
        # Backtrack through these arrays, starting at the lowest possible final V and using the 
        # corresponding P for that, and construct the optimal CDS from that
        # NOTE: I've noticed that for small sequences, the min V[-1] is not the same as the 
        # final predicted summed overall counts for the cds output... 
        q_L = np.argmin(V[-1]) if fastest else np.argmax(V[-1])
        p_i = P[-1][q_L]
        cds = "".join(Q[-1][q_L])
        for i in range(-(L - (self.max_codon - self.min_codon)), -1)[::-1]:
            q_i = Q[i][p_i]
            p_i = P[i][p_i]
            cds = q_i[0] + cds
        if sanity_check:
            return cds, min(V[-1]) if fastest else max(V[-1])
        return cds
    
    def mutate_seq(self, seq, n_mut, n_iter):
        """Given a sequence, return n_iter sequences that code for the same protein
        but differ at <= n_mut codons. Usually, you'll want to use this on optimized 
        sequences to generate slightly sub-optimal sequences. Each iteration mutates
        the mutated sequence from the previous iteration.

        Args:
            seq (str): CDS you want to mutate.
            n_mut (int): How many synonymous codon alterations to make per iteration
            n_iter (int): How many times to repeat mutating the sequence.

        Returns:
            list: A list of mutated sequences.
        """        
        sequences = [seq]
        cod2aa = self.get_codon_to_aa()
        let2cod = self.get_aa_to_codon()

        for _ in range(n_iter):
            seq_current = sequences[-1]
            codons = [seq_current[i:i+3] for i in range(0, len(seq_current), 3)]
            mut_indices = np.random.choice(range(len(codons)), n_mut)
            for i in mut_indices:
                codons[i] = np.random.choice(let2cod[cod2aa[codons[i]]])
            newseq = "".join(codons)
            sequences.append(newseq)
        return sequences[1:]

    @classmethod
    def encode(cls, val: str, ref: dict):
        """
        Encodes a nucleotide or codon value as a one-hot vector 
        based on the provided reference dictionary.

        Args:
            val (str): The value to be encoded (e.g., nucleotide or codon).
            ref (dict): A dictionary mapping values to indices.

        Returns:
            np.ndarray: A one-hot encoded vector of the input value.
            Note that if the input value is not in the provided dictionary, 
            will return a 0 vector.
        """    
        output = np.zeros(len(ref))
        if val in ref:
            output[ref[val]] = 1
        return output
    
    # A bunch of class methods to get variables we can use to encode 
    # nucleotide and codon data for the model
    @classmethod
    def get_codons(cls):
        # List of all possible codons
        alpha = "ACGT"
        return [x + y + z for x in alpha for y in alpha for z in alpha]
    
    @classmethod
    def get_codon_to_id(cls):
        # Codon to ID mapping
        return {codon: idx for idx, codon in enumerate(cls.get_codons())}
    
    @classmethod
    def get_id_to_codon(cls):
        # ID to Codon mapping
        return {idx: codon for codon, idx in cls.get_codon_to_id().items()}
    
    @classmethod
    def get_nt_to_id(cls):
        # Nucleotide to ID mapping
        nts = ["A", "C", "G", "T"]
        return {nt: idx for idx, nt in enumerate(nts)}
    
    @classmethod
    def get_id_to_nt(cls):
        # ID to Nucleotide mapping
        return {idx: nt for nt, idx in cls.get_nt_to_id().items()}

    @classmethod
    def get_aa_to_codon(cls):
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
            '*': ['TAG', 'TGA', 'TAA'],
        }
        return let2cod

    @classmethod
    def get_codon_to_aa(cls):
        """Returns a dictionary of {codon : amino acid}
        """        
        let2cod = cls.get_aa_to_codon()
        return {codon:letter for letter in let2cod for codon in let2cod[letter]}

# USEFUL FUNCTIONS; NOT NEEDED IN THIS SCRIPT
def load_ixnos(pklpath, **kwargs):
    """Loads a .pkl iXnos model into the pytorch implementation.
    Use this if you need to load a model trained using the original paper's scripts.

    Args:
        pklpath (str): path to pkl file with model weights and biases.

    Returns:
        iXnos model
    """    
    model = iXnos(**kwargs)#.to(device)

    with open(pklpath, 'rb') as file:
        # Load the pickled data
        data = pickle.load(file, encoding='bytes')
    for idx, val in enumerate(data):
        data[idx] = torch.from_numpy(val).T
    layer_name = "layers"
    labels = [
        f"{layer_name}.0.weight", f"{layer_name}.0.bias", 
        f"{layer_name}.2.weight", f"{layer_name}.2.bias"]

    old_model = OrderedDict(zip(labels, data))

    model.load_state_dict(old_model)
    return model

def legend_kwargs():
    # Commonly used kwargs for figure legends
    kwargs = {
        "frameon" : False, 
        "bbox_to_anchor" : (1, 0.5), 
        "loc" : "center left"}
    return kwargs

def generate_leaveout_plots(model_dir, min_codon, max_codon, return_comps=False):
    """Generates loss by epoch and correlation by epoch plots, as well as 
    barplots showing the overall difference in pearson correlation between each
    leaveout model and the full model.  

    Args:
        model_dir (str): Directory with full + leaveout models saved. 
        min_codon (int): Minimum codon used in iXnos model. 
        max_codon (int): Maximum codon used in iXnos model. 
        return_comps (bool): Optional argument to return a dictionary of 
        the changes in pearson correlation between each leaveout model 
        and the full model. Defaults to False.

    Returns:
        comps (optionally): a dictionary of the changes in pearson 
        correlation between each leaveout model and the full model.
        Useful in case you want to tweak the final bar plot. 
    """    
    full_df = pd.read_csv(f"{model_dir}/ixnos_n{np.abs(min_codon)}p{max_codon}_full_loss_by_epoch.csv", index_col=0)
    r_full = full_df.iloc[-1]["corr_te"].item()

    named_sites = {
        0: "A",
        -1: "P",
        -2: "E"
    }

    fig0, ax0 = plt.subplots()
    ax0.set_title("Training Loss")
    ax0.set_xlabel("Training Loss")
    fig1, ax1 = plt.subplots()
    ax1.set_title("Test Loss")
    ax1.set_xlabel("Test Loss")
    fig2, ax2 = plt.subplots()
    ax2.set_title("Pearson Correlation vs. Epoch")
    ax2.set_xlabel("Pearson Correlation")

    sns.lineplot(full_df, x = range(len(full_df)), y="loss_tr", ax=ax0, label="Full Model")
    sns.lineplot(full_df, x = range(len(full_df)), y="loss_te", ax=ax1, label="Full Model")
    sns.lineplot(full_df, x = range(len(full_df)), y="corr_te", ax=ax2, label="Full Model")

    comps = {}
    for l in range(min_codon, max_codon+1):
        res = pd.read_csv(f"{model_dir}/ixnos_n{np.abs(min_codon)}p{max_codon}_leaveout_{l}_loss_by_epoch.csv", index_col=0)
        if l in named_sites.keys():
            l = named_sites[l]
        sns.lineplot(res, x = range(len(res)), y="loss_tr", ax=ax0, label=f"Leave out {l}")
        sns.lineplot(res, x = range(len(res)), y="loss_te", ax=ax1, label=f"Leave out {l}")
        sns.lineplot(res, x = range(len(res)), y="corr_te", ax=ax2, label=f"Leave out {l}")
        r_leaveout = res.iloc[-1]["corr_te"].item()
        delta_r = r_full - r_leaveout
        comps[l] = delta_r

    for ax in [ax0, ax1, ax2]:
        ax.legend(title="Model", **legend_kwargs())
        ax.set_xlabel("Epoch")
    plt.show()

    sns.barplot(comps)
    plt.ylabel("\u0394 Correlation")
    plt.xlabel("Codon Site")
    plt.show()
    if return_comps:
        return comps

# FUNCTIONS NECESSARY ONLY FOR THIS SCRIPT
def load_gdf(ydf, gdf_path):
    # Loads the dataframe with gene sequences and finds the (truncated) 
    # CDS we want to feed to iXnos within that sequence
    records = list(SeqIO.parse(gdf_path, "fasta"))
    gdf = pd.DataFrame({
            'ID': [record.id for record in records],
            'seq': [str(record.seq) for record in records]
        }).set_index("ID")
    genes_from_codons = pd.DataFrame(ydf.groupby("gene")["cod_seq"].agg(''.join))
    gdf.loc[genes_from_codons.index, "cod_gene"] = genes_from_codons["cod_seq"].values
    gdf = gdf.dropna()
    assert all(gdf.apply(lambda row: row["cod_gene"] in row["seq"], axis=1))
    gdf["start"] = gdf.apply(lambda row: row['seq'].find(row['cod_gene']), axis=1)
    return gdf


def get_inputs_from_gdf(row, leaveout=None):
    # Get input vectors for a given row of the y dataframe.
    # leaveout: index of codon to leave out in model
    gdf_gene = gdf.loc[row["gene"]]
    gdf_seq = gdf_gene["seq"]
    # Get index of E site in fasta sequence
    esite_index = (row["cod_idx"] - 20) * 3 + gdf_gene["start"] 
    # Find the nucleotides in the window
    footprint_nt = gdf_seq[
        esite_index + NT_INDICES[0]:
        esite_index + NT_INDICES[-1] + 1]
    if leaveout is not None:
        l_start = 3 * (leaveout) - max(NT_INDICES) - 1
        l_stop = l_start + 3
        footprint_nt = footprint_nt[: l_start] + footprint_nt[l_stop:] if l_stop != 0 else footprint_nt[: l_start]
    # Convert nt string into codons 
        # NOTE: assumes your nt footprint is an in frame CDS containing just the codons we wanna look at
    footprint_codons = [footprint_nt[i:i+3] for i in range(0, len(footprint_nt), 3)]
    # Assemble input tensor
    codon_vector = np.concatenate([iXnos.encode(i, COD2ID) for i in footprint_codons])
    nt_vector = np.concatenate([iXnos.encode(i, NT2ID) for i in footprint_nt])
    input_vector = np.concatenate([codon_vector, nt_vector])
    input_vector = torch.from_numpy(input_vector).to(torch.float32)
    return input_vector

def train_loop(dataloader, model, loss_fn, optimizer):
    # Training loop for iXnos model
    model.train()
    total_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()  # Zero the gradients
        outputs = model(X)  # Forward pass
        outputs = outputs.squeeze(1) # Squeeze outputs to be right dims for comparison to y w loss fn
        loss = loss_fn(outputs, y)  # Compute the loss
        total_loss += loss.item()
        loss.backward()  # Backward pass (compute gradients)
        optimizer.step() # Update parameters using Nesterov SGD
    scheduler.step() # Update learning rate using scheduler
    avg_loss = total_loss / len(dataloader)
    current_lr = scheduler.get_last_lr()[0]
    print(f"Average Loss: {avg_loss} | Learning Rate: {current_lr:.6f}", end = " ")
    return avg_loss

def test_loop(dataloader, model, loss_fn):
    # Test loop for iXnos model
    model.eval()
    predicted_counts, actual_counts = [], []
    total_loss = 0
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            # Predict scaled counts
            outputs = model(X)
            outputs = outputs.squeeze(1)
            predicted_counts.append(outputs.cpu().numpy())
            actual_counts.append(y.cpu().numpy())
            # Compute loss
            loss = loss_fn(outputs, y)
            total_loss += loss.item()  # Accumulate the loss for this batch

        predicted_counts = np.concatenate(predicted_counts)
        actual_counts = np.concatenate(actual_counts)
        # Find pearson correlation between predictions and actual
        pearson_r, _ = pearsonr(predicted_counts, actual_counts)
        avg_loss = total_loss / len(dataloader)
    print(f"| Pearson Correlation: {pearson_r:.4f} | Test Loss: {avg_loss:.4f}")
    return avg_loss, pearson_r

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="General script to train iXnos model on preprocessed ribo-seq data")
    parser.add_argument(
        '-d', '--data_dir', type=str, required=True, 
        help="Path to the data directory (NOTE: actual data files must be in a sub-directory named \"process\")"
    )
    parser.add_argument(
        '-o', '--output_dir', type=str, required=True, 
        help="Path to the output directory (NOTE: actual data files must be in a sub-directory named \"process\")"
    )
    parser.add_argument(
        '-l', '--leaveout', type=int, default=None, 
        help="Codon index to leave out (optional)"
    )
    parser.add_argument(
        '-g', '--gdf_path', type=str, required=True, 
        help="Path to transcriptome fasta file"
    )
    parser.add_argument(
        '-n', '--pos_5', type=int, default=-5, 
        help="5' Codon Position. Defaults to -5. (NOTE: should be negative. If you provide a positive value it will be converted)"
    )
    parser.add_argument(
        '-p', '--pos_3', type=int, default=4, 
        help="3' Codon Position. Defaults to +4. (NOTE: should be positive)"
    )
    args = parser.parse_args()
    data_dir = args.data_dir
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    leaveout = args.leaveout
    gdf_path = args.gdf_path#"../iXnos/genome_data/human.transcripts.13cds10.transcripts.fa"
    pos_5 = args.pos_5
    pos_5 = pos_5 if pos_5 < 0 else -pos_5
    pos_3 = args.pos_3
    print(f"Training an iXnos model\n\tCodon range: {pos_5} to {pos_3}\n\tLeaveout index: {leaveout}")
    # Variables for encoding inputs
    CODONS = iXnos.get_codons()  # List of all possible codons
    COD2ID = iXnos.get_codon_to_id()  # Codon to ID mapping
    ID2COD = iXnos.get_id_to_codon()  # ID to Codon mapping
    NT2ID = iXnos.get_nt_to_id()  # Nucleotide to ID mapping
    ID2NT = iXnos.get_id_to_nt()  # ID to Nucleotide mapping  
    # Initialize model
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")
    # Initialize model
    model = iXnos(min_codon = pos_5, max_codon = pos_3, leaveout=leaveout).to(device)
    CODON_INDICES = model.codon_indices  # Codon indices window
    # NOTE: I'm NOT using the iXnos nt_indices property here just to be compatible with the get_inputs_from_gdf function
    NT_INDICES = np.arange(3*pos_5, 3*(pos_3 + 1))  # Nucleotide indices window
    # Read the pre-calculated train and test datasets as pandas dataframes
    # Detect train and test datasets
    train_naming_scheme = f"{data_dir}/tr_*data_table.txt"
    files_tr = glob.glob(train_naming_scheme)
    filepath_train = files_tr[0]
    print(f"Training set: found {files_tr}, using {filepath_train}")
    
    test_naming_scheme = f"{data_dir}/te_*data_table.txt"
    files_te = glob.glob(test_naming_scheme)
    filepath_test = files_te[0]
    print(f"Test set: found {files_te}, using {filepath_test}")

    ydf_te = pd.read_csv(
        filepath_test,
        sep='\t')
    ydf_tr = pd.read_csv(
        filepath_train,
        sep='\t')
    # Get the train and test genes
    genes_te = ydf_te["gene"].unique()
    genes_tr = ydf_tr["gene"].unique()
    # Combine these dataframes and encode each codon
    ydf = pd.concat([ydf_te, ydf_tr]).sort_values(by=["gene", "cod_idx"])
    gdf = load_gdf(ydf, gdf_path)
    ydf["X"] = ydf.apply(get_inputs_from_gdf, axis=1, leaveout=leaveout)
    # Extract inputs and outputs for model
    X_tr, y_tr = ydf.loc[ydf["gene"].isin(genes_tr), ["X", "scaled_cts"]].values.T.tolist()
    X_te, y_te = ydf.loc[ydf["gene"].isin(genes_te), ["X", "scaled_cts"]].values.T.tolist()
    X_tr = torch.stack(X_tr)
    y_tr = torch.tensor(y_tr)
    X_te = torch.stack(X_te)
    y_te = torch.tensor(y_te)
    # Parameters for training
    epochs = 70
    learning_rate = 0.01
    lr_decay = 32
    batch_size = 500
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)
    # LR decay function
    lr_lambda = lambda epoch: 1 / (1 + float(epoch) / lr_decay)
    # LambdaLR scheduler: Use lr_lambda to apply custom decay
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    # Prepare data for model
    dataset_tr = TensorDataset(X_tr, y_tr)
    train_loader = DataLoader(dataset_tr, batch_size=batch_size, shuffle=True)
    dataset_te = TensorDataset(X_te, y_te)
    test_loader = DataLoader(dataset_te, batch_size=batch_size, shuffle=True)
    # Lists to store loss + correlations as training goes on
    loss_tr, loss_te, corr_te = [np.nan], [], [] 
    # Evaluate the model before any training
    print("Initial", end=" ")
    l_te_0, r_te_0 = test_loop(test_loader, model, loss_fn)
    loss_te.append(l_te_0)
    corr_te.append(r_te_0)
    # Train the model
    for epoch in range(epochs):
        print(f"Epoch {epoch}:", end = " ")
        l_tr = train_loop(train_loader, model, loss_fn, optimizer)
        loss_tr.append(l_tr)
        # if (epoch + 1) % 5 == 0:
        l_te, r_te = test_loop(test_loader, model, loss_fn)
        loss_te.append(l_te)
        corr_te.append(r_te)
    # Save the model
    if leaveout is not None:
        model_name = f'ixnos_n{-pos_5}p{pos_3}_leaveout_{leaveout}'
    else:
        model_name = f'ixnos_n{-pos_5}p{pos_3}_full'
    torch.save(model.state_dict(), f'{output_dir}/{model_name}.pth')
    # Save the stored loss values + pearson correlations
    df_res = pd.DataFrame({
        "loss_tr": loss_tr, 
        "loss_te": loss_te, 
        "corr_te": corr_te,})
    df_res.index.name = "epoch"
    df_res.to_csv(f'{output_dir}/{model_name}_loss_by_epoch.csv')
