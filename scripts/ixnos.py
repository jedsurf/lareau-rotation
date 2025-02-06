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

# GENERAL SCRIPT to run iXnos on ribo-seq data
# Sample linear leaveout series command:
    # for i in {-5..4}; do python ixnos.py -d ../processed-data/thp1 -o ../processed-data/thp1/models -g ../iXnos/genome_data/human.transcripts.13cds10.transcripts.fa -l $i > ../processed-data/thp1/logs/leaveout_$i.log 2>&1; done

class iXnos(nn.Module):
    def __init__(self, n_codons = 10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_codons = n_codons
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
   
# class iXnosLeaveOut(nn.Module):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.layers = nn.Sequential(
#             nn.Linear(684, 200),
#             nn.Tanh(),
#             nn.Linear(200, 1),
#             nn.ReLU(),
#         )
    
#     def forward(self, x):
#         output = self.layers(x)
#         return output 

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

def encode(val: str, ref: dict):
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

def get_inputs(row, leaveout=None):
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
    codon_vector = np.concatenate([encode(i, COD2ID) for i in footprint_codons])
    nt_vector = np.concatenate([encode(i, NT2ID) for i in footprint_nt])
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
    # ALPHA = "ACGT"  # Defines the possible nucleotides
    # NTS = ["A", "C", "G", "T"]  # Nucleotide list
    CODONS = iXnos.get_codons()  # List of all possible codons
    COD2ID = iXnos.get_codon_to_id()  # Codon to ID mapping
    ID2COD = iXnos.get_id_to_codon()  # ID to Codon mapping
    NT2ID = iXnos.get_nt_to_id()  # Nucleotide to ID mapping
    ID2NT = iXnos.get_id_to_nt()  # ID to Nucleotide mapping  
    # Define the indices of the codons and nucleotides iXnos will use  
    CODON_INDICES = np.arange(pos_5, pos_3 + 1)  # Codon indices window
    NT_INDICES = np.arange(3*pos_5, 3*(pos_3 + 1))  # Nucleotide indices window
    N_CODONS = len(CODON_INDICES)
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
    if leaveout is None:
        model = iXnos(n_codons=N_CODONS).to(device)
    else:
        model = iXnos(n_codons=N_CODONS - 1).to(device)
    # Read the pre-calculated train and test datasets as pandas dataframes
    # Detect train and test datasets
    train_naming_scheme = f"{data_dir}/process/tr_set_bounds*data_table.txt"
    files_tr = glob.glob(train_naming_scheme)
    filepath_train = files_tr[0]
    print(f"Training set: found {files_tr}, using {filepath_train}")
    
    test_naming_scheme = f"{data_dir}/process/te_set_bounds*data_table.txt"
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
    ydf["X"] = ydf.apply(get_inputs, axis=1, leaveout=leaveout)
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
