import xml.etree.ElementTree as ET
import pandas as pd
import io

#df = pd.read_csv('PubChem_compound_cache_gMclmVZBM_0E1zHOs7Z46XhcQTztAo029xOWeuwChHvsG7g.csv', sep=',')

#xml_data = ET.parse('PubChem_compound_cache_ZiHDf7Im15rgsNWpV9Gcjpw6QVqHrpPO6euIgvL6moPy46Y.xml')

#xml_content = xml_data

def parsear_farmacos_simple():
    print("--- INICIANDO PARSEO XML (Estructura PubChem Personalizada) ---")

    tree = ET.parse('PubChem_compound_cache_ZiHDf7Im15rgsNWpV9Gcjpw6QVqHrpPO6euIgvL6moPy46Y.xml')
    root = tree.getroot()

    farmacos_extraidos = []

    # Iteramos sobre cada etiqueta <row>
    for drug in root.findall('row'):
        data = {}
        
        # --- 1. Extraer ID y Nombre ---
        # Usamos una función auxiliar o comprobación inline para evitar errores
        # si el campo (como annotation) no existe en alguna fila.
        
        cid_elem = drug.find('cid')
        data['cid'] = cid_elem.text if cid_elem is not None else "N/A"
        name_elem = drug.find('cmpdname')
        data['cmpdname'] = name_elem.text if name_elem is not None else "Sin Nombre"
        smiles_elem = drug.find('smiles')
        data['smiles'] = smiles_elem.text if smiles_elem is not None else ""
        
        annot_elem = drug.find('annotation')
        annotations_list = []
        
        if annot_elem is not None:
            # Paso A: Buscar si tiene hijos <sub-annotation>
            sub_annots = annot_elem.findall('sub-annotation')
            
            if len(sub_annots) > 0:
                # ES UNA LISTA: Iteramos sobre los hijos
                for sub in sub_annots:
                    if sub.text:
                        annotations_list.append(sub.text)
            else:
                # ES TEXTO PLANO: Tomamos el texto directo si existe
                if annot_elem.text and annot_elem.text.strip():
                    annotations_list.append(annot_elem.text.strip())
        
        # Guardamos siempre como lista para ser consistentes
        data['annotation'] = annotations_list if annotations_list else "Sin Anotación"

        # --- 2. Extraer Atributos Extra (Tu lógica) ---
        xtra_attributes = []
        # Nota: 'exactmas' parece un typo en tu lista original, pero lo mantengo 
        # junto a 'exactmass' por si acaso.
        xtra_at_elem = ['iupacname', 'exactmass', 'xlogp', 'mf', 'hbonddonor', 'hbondacc']
        
        for attr in xtra_at_elem:
            elem = drug.find(attr)
            if elem is not None and elem.text is not None:
                xtra_attributes.append((attr, elem.text))
        
        data['xtra_attributes'] = xtra_attributes
        
        farmacos_extraidos.append(data)
        
        # --- Imprimir resultado ---
        print(f"✅ Procesado: {data['cmpdname']} (CID: {data['cid']})")
        
        # Manejo seguro del print de SMILES
        smiles_display = data['smiles'][:30] + "..." if len(data['smiles']) > 30 else data['smiles']
        print(f"   SMILES: {smiles_display}")
        
        print(f"   Anotación: {data['annotation'][:50]}") # Cortamos si es muy larga
        
        # Imprimimos los atributos extra encontrados de forma legible
        extras_str = ", ".join([f"{k}={v}" for k, v in data['xtra_attributes']])
        print(f"   Extras: {extras_str}")
        print("-" * 30)

    return farmacos_extraidos

if __name__ == "__main__":
    datos = parsear_farmacos_simple()

    out_df = pd.DataFrame(datos)
    out_df.to_csv('farmacos_extraidos.tsv', sep='\t', index=False)

    print(f"\nTotal extraídos: {len(datos)}")






