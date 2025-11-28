import torch
import torch.nn as nn
import einops
from transformers import AutoTokenizer, AutoModel, AutoConfig
from gen_tokenizer import DNAEncoder

if __name__ == "__main__":
    
    secuencias_test = [
        "ACGTAGCTAGCTAGCTAGCTAGCTAGCGTAGCTAGCT",              
        "ACGTAGCTAGCTAGCTAGCTAGCTAGCGTAGCTAGCA",              
        "ACGT",                                               
        "NNNNNNGCTAGCTAGCTAGCTNNNNN"                          
    ]
    
    batch_size = len(secuencias_test)
    print(f"--- INICIANDO TEST (Modelo GENA-LM Corregido) ---")

    # PRUEBA A: Reducción (768 -> 256)
    target_dim_a = 256
    encoder_a = DNAEncoder(output_dim=target_dim_a)
    output_a = encoder_a(secuencias_test)
    
    print("\n[RESULTADOS TEST A]")
    print(f"Output Real: {output_a.shape}")
    
    if output_a.shape == (batch_size, target_dim_a):
        print("✅ PRUEBA A SUPERADA")
        print(f"Vector sample: {output_a[0][:5].tolist()}...")
    else:
        print("❌ PRUEBA A FALLIDA")

    # PRUEBA B: Integridad
    if not torch.isnan(output_a).any():
        print("✅ PRUEBA B SUPERADA (Sin NaNs)")
    else:
        print("❌ PRUEBA B FALLIDA")

    print("\n--- FIN ---")