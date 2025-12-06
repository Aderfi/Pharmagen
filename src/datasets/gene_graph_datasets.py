"""
gene_graph_datasets.py

Handles the generation of genomic embeddings using Nucleotide Transformers.
Extracts DNA sequences based on genomic coordinates and computes embeddings.

Adheres to: Zen of Python, SOLID.
"""

import logging
from typing import Optional, Tuple
import torch
from torch.utils.data import Dataset
from Bio import SeqIO
from transformers import AutoTokenizer, AutoModelForMaskedLM

logger = logging.getLogger(__name__)

# Default Model
MODEL_NAME = " InstaDeepAI/nucleotide-transformer-500m-human-ref" 
# Alternative: "zhihan1996/DNABERT-2-117M" as seen in modeling.py context.
# I'll stick to a generic logic where model_name can be passed.

class GeneEmbeddingDataset(Dataset):
    """
    Dataset to compute embeddings for genomic regions on the fly (or cached).
    """
    def __init__(
        self, 
        data_source, # DataFrame or list of dicts with coords
        ref_genome_path: str,
        gtf_path: str, # Context implies using GTF for something, maybe gene lookup?
        model_name: str = "zhihan1996/DNABERT-2-117M",
        window_size: int = 512,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.data_source = data_source
        self.ref_genome_path = ref_genome_path
        self.gtf_path = gtf_path
        self.window_size = window_size
        self.device = device
        
        # Load Genome (Lazy loading might be better for huge files, using dict for now)
        logger.info(f"Loading Reference Genome from {ref_genome_path}...")
        # Note: For full human genome, SeqIO.to_dict is memory intensive. 
        # In production, use Fasta index (FastaFile). Assuming Biopython handles it.
        self.genome_dict = SeqIO.to_dict(SeqIO.parse(ref_genome_path, "fasta"))
        logger.info("Genome loaded.")

        # Load Model & Tokenizer
        logger.info(f"Loading Nucleotide Transformer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True).to(self.device)
        self.model.eval()

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        # Assuming data_source has columns/keys: 'chrom', 'pos'
        item = self.data_source[idx]
        chrom = item['chrom']
        pos = int(item['pos']) # 1-based absolute position?
        
        sequence = self._get_sequence(chrom, pos)
        return self._compute_embedding(sequence)

    def _get_sequence(self, chrom: str, pos: int) -> str:
        """Extracts sequence from loaded genome dict."""
        # We take a window centered on 'pos'
        start = max(0, pos - (self.window_size // 2))
        end = start + self.window_size
        
        if chrom not in self.genome_dict:
            # logger.error(f"Chromosome {chrom} not found.") # Too noisy
            return "N" * self.window_size # Return dummy

        seq_record = self.genome_dict[chrom]
        end = min(len(seq_record), end)
        
        dna_sequence = str(seq_record.seq[start:end]).upper()
        
        if len(dna_sequence) < self.window_size:
            pad_len = self.window_size - len(dna_sequence)
            dna_sequence += "N" * pad_len
            
        return dna_sequence

    def get_embedding_for_coords(self, chrom: str, pos: int) -> torch.Tensor:
        """Public API to get embedding for specific coordinates."""
        seq = self._get_sequence(chrom, pos)
        return self._compute_embedding(seq)

    def _compute_embedding(self, sequence: str) -> torch.Tensor:
        """Computes the embedding for a DNA sequence."""
        inputs = self.tokenizer(
            sequence, 
            return_tensors="pt", 
            padding="max_length", 
            truncation=True, 
            max_length=self.window_size
        )
        
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            # Use the last hidden state, mean pooling or CLS
            # DNABERT-2 / Nucleotide Transformer strategies vary. 
            # Usually last hidden state average is safe.
            hidden_states = outputs.hidden_states[-1] # [Batch, Seq_Len, Emb_Dim]
            
            # Mean Pooling (ignoring padding)
            # mask expanded: [Batch, Seq_Len, 1]
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            embedding = sum_embeddings / sum_mask
            
        return embedding.squeeze(0).cpu()

