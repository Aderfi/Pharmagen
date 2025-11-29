import re
import pubchempy as pcp
import rdkit
import pandas as pd
import json
from tqdm import tqdm
import time
import polars as pl
from src.d_graphs.smiles_to_graph import smiles_to_graph_complete
import pickle
import gzip
import networkx as nx

def main():
    df = pl.read_csv('Drug_Compount_processed.tsv', separator='\t', has_header=True)
    
    df_dict = {k: v for k,v in zip(df['Compound_CID'].to_list(), df['SMILES'].to_list())}

    with gzip.open('drug_graphs.pkl.gz', 'wb') as f:
        #results = {}
        for cid, smiles in tqdm(df_dict.items(), total=len(df_dict), desc="Processing SMILES to Graphs"):
            graph_data = smiles_to_graph_complete(smiles)
            if graph_data is not None:
                # Convert torch tensors to lists for JSON serialization
                graph_dict = {
                    'x': graph_data.x.tolist(),
                    'edge_index': graph_data.edge_index.tolist(),
                    'edge_attr': graph_data.edge_attr.tolist() if graph_data.edge_attr is not None else None
                }
                #results[cid] = graph_dict
                temp = {cid: graph_dict}
                pickle.dump(temp, f)         

if __name__ == "__main__":
    main()     