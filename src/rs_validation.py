import Bio.phenotype
import xml.dom.minidom as xld
from Bio import Entrez
Entrez.email = "zeooloo@gmail.com"  # Proporciona tu correo electrónico aquí

handle = Entrez.efetch(db="snp", id="1065852", retmode="xml")

data = handle.read()
print(data)