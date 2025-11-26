import pubchempy as pcp
import time
from pubchempy import Compound
from typing import List

"""
drug_names: List[str] = ["aspirin", "acetaminophen", "4-OHtramadol, 4-hydroxytramadol, SN-38"]


compounds = pcp.get_compounds(drug_names, 'name', listkey_count=100, as_dataframe=False) # type: ignore[arg-type]
time.sleep(3)  # To respect PubChem's rate limiting policy

print("CIDs retrieved:", [compound.cid for compound in compounds]) # type: ignore[attr-defined]
"""

drug_names = ["aspirin", "acetaminophen", "4-OHtramadol, 4-hydroxytramadol, SN-38"]
out_cids = []
"""
for i in drug_names:
    name = i # type: ignore
    cid = pcp.get_substances(name, 'name') # type: ignore[index]
    out_cids.append(cid)
    time.sleep(3)  # To respect PubChem's rate limiting policy
"""

name = "SN-38"

df = pcp.get_substances(name, 'name') # type: ignore[index]

with open("pubchem_output.txt", "w") as f:
    for line in df:
        f.write(str(line) + "\n")
print("CIDs retrieved:", df)