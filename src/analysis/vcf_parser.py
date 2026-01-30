# Pharmagen - VCF to Model Bridge
# Copyright (C) 2025 Adrim Hamed Outmani

import pandas as pd
import pysam
from pathlib import Path
from typing import Any

class VCFParser:
    """
    Bridge component that transforms raw VCF files into structured DataFrames
    ready for the Pharmagen DeepFM inference engine.
    """
    def __init__(self, vcf_path: Path):
        self.vcf_path = vcf_path
        if not self.vcf_path.exists():
            raise FileNotFoundError(f"VCF file not found: {vcf_path}")

    def parse_to_dataframe(self, sample_id: str | None = None) -> pd.DataFrame:
        """
        Parses the VCF and returns a clean DataFrame with columns expected by the model.
        
        Columns generated:
          - Variant_Normalized (e.g., "rs123:AT")
          - Variant_ID (e.g., "rs123")
          - Gene_Symbol (Extracted if annotated, else Chromosome)
          - Allele (e.g., "A/T")
          - Genotype_Type (Heterozygous, Homozygous Alt)
        """
        data_rows = []
        
        with pysam.VariantFile(str(self.vcf_path)) as vcf:
            # Auto-detect sample ID if not provided
            if not sample_id:
                samples = list(vcf.header.samples)
                if not samples:
                    raise ValueError("No samples found in VCF header.")
                sample_id = samples[0] # Default to first sample
            
            # Iterate variants
            for record in vcf:
                # Skip filtered or low quality if needed (configurable?)
                # if record.filter.keys() and "PASS" not in record.filter.keys(): continue
                
                # 1. Decode Genotype
                gt = self._decode_genotype(record, sample_id)
                if not gt: 
                    continue # Skip non-called (./.)

                # 2. Extract Identifier (Prefer rsID, fallback to chr:pos)
                var_id = record.id if record.id else f"{record.chrom}:{record.pos}"
                
                # 3. Construct Normalized Feature
                # Example format: "rs123:AG" (ID + Genotype String)
                variant_normalized = f"{var_id}:{gt['alelos_clean']}"
                
                # 4. Extract Gene (If annotated by VEP/SnpEff in INFO)
                # This depends heavily on your annotation pipeline.
                # Standard VEP field is usually 'CSQ' or 'ANN'.
                gene_symbol = self._extract_gene_symbol(record)

                row = {
                    "Variant_Normalized": variant_normalized,
                    "Variant/Haplotypes": var_id,   # Mapping to training col name
                    "Gene_Symbol": gene_symbol,
                    "Allele": gt['alelos_slash'],   # "A/G"
                    "Zygosity": gt['tipo'],         # "Heterozygous"
                    "Chrom": record.chrom,
                    "Pos": record.pos
                }
                data_rows.append(row)
                
        return pd.DataFrame(data_rows)

    def _decode_genotype(self, record, sample_id) -> dict[str, Any] | None:
        """
        Extracts genotype info safely using Pysam.
        """
        call = record.samples[sample_id]
        gt_tuple = call['GT'] # e.g. (0, 1)

        if None in gt_tuple:
            return None # Missing call

        # Decode bases (0=Ref, 1=Alt1, ...)
        bases = []
        for idx in gt_tuple:
            if idx == 0:
                bases.append(record.ref)
            else:
                # idx-1 because record.alts is a tuple of alts only
                bases.append(record.alts[idx-1])
        
        # Strings
        alelos_slash = "/".join(bases)      # "A/G"
        alelos_clean = "".join(bases)       # "AG" (Useful for embeddings)

        # Type logic
        if len(set(gt_tuple)) == 1:
            tipo = "Homozygous Reference" if gt_tuple[0] == 0 else "Homozygous Alt"
        else:
            tipo = "Heterozygous"

        return {
            "alelos_slash": alelos_slash,
            "alelos_clean": alelos_clean,
            "tipo": tipo
        }

    def _extract_gene_symbol(self, record) -> str:
        """
        Attempts to find Gene Symbol from VEP/SnpEff annotations (INFO field).
        Falls back to 'Unknown'.
        """
        # Common VEP key is 'CSQ'
        if 'CSQ' in record.info:
            # CSQ format is defined in header, usually: Allele|Consequence|IMPACT|SYMBOL|...
            # This is a simplification; robust parsing requires reading header.
            # Assuming SYMBOL is often the 3rd or 4th field.
            csq_entries = record.info['CSQ']
            if csq_entries:
                # Take first transcript
                first_csq = csq_entries[0].split('|') # noqa
                # HACK: Iterate fields to find something looking like a gene symbol?
                # Better: Read VCF header for CSQ format definition.
                # For now, return placeholder or try to fish it if known index.
                pass 
        
        return "Unknown"

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        vcf_in = Path(sys.argv[1])
        parser = VCFParser(vcf_in)
        df = parser.parse_to_dataframe()
        print(f"Extracted {len(df)} variants.")
        print(df.head())
        
        # Save for inspection
        df.to_csv(vcf_in.with_suffix(".tsv"), sep="\t", index=False)
            
