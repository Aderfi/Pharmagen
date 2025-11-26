import time
import re
import pubchempy as pcp
import pandas as pd
import requests

with open('drug_names.txt', 'r') as file: 
    drug_list = [line.strip() for line in file.readlines()]

def search_drug_cascade(drug_name):
    # --- LEVEL 1: Standard Compound Search ---
    try:
        compounds = pcp.get_compounds(drug_name, 'name')
        if compounds:
            cmpd = compounds[0]
            return {
                'Input Name': drug_name,
                'PubChem Name': cmpd.iupac_name, # or cmpd.synonyms[0] if available
                'CID': cmpd.cid,
                'Molecular Formula': cmpd.molecular_formula,
                'Molecular Weight': cmpd.molecular_weight,
                'SMILES_code': cmpd.smiles,
                'SMILES_connectivity': cmpd.connectivity_smiles,
                'Status': 'Found in Compounds'
            }
    except Exception:
        pass

    # --- LEVEL 2: Substance Search (For Biologics like Abciximab) ---
    # If Level 1 failed, we search the "Substance" database.
    try:
        substances = pcp.get_substances(drug_name, 'name')
        if substances:
            # We take the first result, but note that Substances are "messier"
            sub = substances[0]
            
            # Sometimes a Substance is linked to a Compound (standardized_cid)
            linked_cid = sub.standardized_cid
            
            if linked_cid:
                # If there is a link, fetch the compound data
                cmpd = pcp.Compound.from_cid(linked_cid)
                return {
                    'Input Name': drug_name,
                    'PubChem Name': cmpd.iupac_name, # or cmpd.synonyms[0] if available
                    'CID': cmpd.cid,
                    'Molecular Formula': cmpd.molecular_formula,
                    'Molecular Weight': cmpd.molecular_weight,
                    'SMILES_code': cmpd.smiles,
                    'SMILES_connectivity': cmpd.connectivity_smiles,
                    'Status': 'Found in Subs'
                }
            else:
                # If NO link, it is likely a true Biologic/Complex Substance
                return {
                    'Name': drug_name,
                    'Type': 'Biologic/Complex',
                    'sID': f"{sub.sid}",
                    'Mol_Weight': "N/A (Biologic)",
                    'LogP': "N/A (Biologic)",
                    'Status': 'Found in Substances (No Structure)'
                }
    except Exception:
        pass

    # --- LEVEL 3: Autocomplete/Spell Check (For Typos) ---
    # PubChem has a hidden API for autocomplete that handles typos better than pcp
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/autocomplete/compound/{drug_name}/json?limit=1"
        response = requests.get(url).json()
        if response['total'] > 0:
            suggestion = response['dictionary_terms']['compound'][0]
            if suggestion.lower() != drug_name.lower():
                # Found a suggestion, recurse with the corrected name
                result = search_drug_cascade(suggestion)
                result['Notes'] = f"Auto-corrected from {drug_name}"
                return result
    except Exception:
        pass

    # --- LEVEL 4: Give Up ---
    return {
        'Name': drug_name,
        'Type': 'Unknown',
        'ID': None,
        'Mol_Weight': None,
        'LogP': None,
        'Status': 'Not Found'
    }

# Run the cascade
results = []
print("Starting Cascade Search...")

for drug in drug_list:
    print(f"Searching for {drug}...")
    data = search_drug_cascade(drug)
    results.append(data)
    time.sleep(0.3) # Respect API limits

# Display
df = pd.DataFrame(results)
df.to_csv('normalized_drugs_output.tsv', index=False, sep='\t')
