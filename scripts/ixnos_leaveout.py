import torch
from torch import nn
import pickle
from collections import OrderedDict
import numpy as np
from Bio import SeqIO
import pandas as pd
import sys
from scipy.stats import pearsonr
from torch.utils.data import DataLoader, TensorDataset

class iXnosLeaveOut(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layers = nn.Sequential(
            nn.Linear(684, 200),
            nn.Tanh(),
            nn.Linear(200, 1),
            nn.ReLU(),
        )
    
    def forward(self, x):
        output = self.layers(x)
        return output 

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
    """    
    output = np.zeros(len(ref))
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
    leaveout = int(sys.argv[1])
    print(f"Running leaveout model without codon {leaveout}")
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")
    # Initialize model
    model = iXnosLeaveOut().to(device)
    # Variables for encoding inputs
    ALPHA = "ACGT"  # Defines the possible nucleotides
    NTS = ["A", "C", "G", "T"]  # Nucleotide list
    CODONS = [x + y + z for x in ALPHA for y in ALPHA for z in ALPHA]
    COD2ID = {codon: idx for idx, codon in enumerate(CODONS)}  # Codon to ID mapping
    ID2COD = {idx: codon for codon, idx in COD2ID.items()}  # ID to Codon mapping
    NT2ID = {nt: idx for idx, nt in enumerate(NTS)}  # Nucleotide to ID mapping
    ID2NT = {idx: nt for nt, idx in NT2ID.items()}  # ID to Nucleotide mapping
    NT_INDICES = np.arange(-15, 15)  # Nucleotide indices window
    CODON_INDICES = np.arange(-5, 5)  # Codon indices window
    assert leaveout in CODON_INDICES, \
        f"{leaveout} not a valid codon index! Must be in {CODON_INDICES}"
    # Read the pre-calculated train and test datasets as pandas dataframes
    print("Reading data...")
    iwasaki_dir = "../iXnos/expts/iwasaki"
    ydf_te = pd.read_csv(
        f"{iwasaki_dir}/process/te_set_bounds.size.27.30.trunc.20.20.min_cts.200.min_cod.100.top.500.data_table.txt",
        sep='\t')
    ydf_tr = pd.read_csv(
        f"{iwasaki_dir}/process/tr_set_bounds.size.27.30.trunc.20.20.min_cts.200.min_cod.100.top.500.data_table.txt",
        sep='\t')
    # Get the train and test genes
    genes_te = ydf_te["gene"].unique()
    genes_tr = ydf_tr["gene"].unique()
    # Combine these dataframes and encode each codon
    ydf = pd.concat([ydf_te, ydf_tr]).sort_values(by=["gene", "cod_idx"])
    gdf_path = "../iXnos/genome_data/human.transcripts.13cds10.transcripts.fa"
    gdf = load_gdf(ydf, gdf_path)
    print("Encoding inputs...")
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
    output_dir = "../models"
    torch.save(model.state_dict(), f'{output_dir}/ixnos_leaveout_{leaveout}.pth')
    # Save the stored loss values + pearson correlations
    df_res = pd.DataFrame({
        "loss_tr": loss_tr, 
        "loss_te": loss_te, 
        "corr_te": corr_te,})
    df_res.index.name = "epoch"
    df_res.to_csv(f'{output_dir}/ixnos_leaveout_{leaveout}_loss_by_epoch.csv')
