import pandas as pd
from thefuzz import process, fuzz
import json
import re

df = pd.read_csv('test_pheno_Pop_var_nofa_mapped_filled.tsv', sep='\t')

with open ('dict_effect-phenotype_id.json', 'r') as f:
    effect_phenotype_dict = json.load(f)
"""
categories_list = [
    'Side Effect',
    'Efficacy',
    'Disease',
    'Other',
    'PK',
]

mask = df['Effect_phenotype'] != 'Fenotipo_No_Especificado'

for idx, field in df['Effect_phenotype'][mask].items():
    
    import re

# ... (asumiendo que df, mask, idx y field están definidos en tu bucle)
# for idx, field in df['Effect_phenotype'][mask].iterrows():
#     ...

    if '"' in field:
        # 1. Usar re.findall para extraer *solo* el contenido de las comillas
        #    Esto evita los problemas de split con cadenas vacías y comas.
        field_groups = re.findall(r'"(.*?)"', field)
        
        final_strings = [] # Una lista para guardar las cadenas procesadas
        
        for group in field_groups:
            # group será "Efficacy:primary graft failure" 
            # o "Efficacy:Kidney Tubular Necrosis, Acute"
            
            try:
                # 2. Separar solo por el PRIMER colon (maxsplit=1)
                cat, inf = group.split(':', 1)
                cat = cat.strip()
                inf = inf.strip()

                # 3. Lógica de inversión (reparada)
                inf_parts = inf.split(', ')
                
                if len(inf_parts) == 2:
                    # Si tiene el formato "Nombre, Tipo" (ej: "Kidney..., Acute")
                    # b = inf_parts[0] ("Kidney Tubular Necrosis")
                    # a = inf_parts[1] ("Acute")
                    # Lo reordenamos como "Tipo Nombre"
                    processed_inf = inf_parts[1].strip() + ' ' + inf_parts[0].strip()
                else:
                    # Si no tiene coma (o tiene más de una), se deja como está
                    processed_inf = inf

                # Añadir la cadena final "Categoría:Info Procesada" a la lista
                final_strings.append(cat + ':' + processed_inf)
            
            except ValueError:
                # Omitir grupos que no tengan el formato "Categoría:Info"
                continue 

        # 4. Unir todas las cadenas procesadas con ", "
        #    Esto crea el formato de string único que pediste.
        df.loc[idx, 'Effect_phenotype'] = ', '.join(final_strings) #type: ignore

    
"""

side_effect_map = {}
for item in effect_phenotype_dict:
    # Almacena el mapeo: "Nombre del efecto" -> "ID"
    # Usamos .lower() para una coincidencia más robusta y sin distinción de mayúsculas
    side_effect_map[item['side_effect'].lower()] = item['id_se']

# Crea la lista de todos los nombres posibles (claves del map) para el fuzzy matching
side_effect_choices = list(side_effect_map.keys())

print(side_effect_choices[:10])  # Muestra las primeras 10 opciones para verificar


# --- CÓDIGO DE MAPEO DE EFECTOS A IDs ---
j=0
list_indexes_not_mapped = []
mask = df['Effect_phenotype'] != 'Fenotipo_No_Especificado'


# 2. Crear una nueva columna para guardar los resultados del mapeo
df['Effect_phenotype_id'] = pd.NA

# 3. Iterar sobre el dataframe (en las filas limpiadas) para mapear IDs
#    Usamos la misma 'mask' que antes para operar solo sobre las filas relevantes.
for idx, row_string in df.loc[mask, 'Effect_phenotype'].items():

    if not isinstance(row_string, str) or not row_string:
        continue

    # La fila puede tener múltiples entradas: "Cat1:Info1, Cat2:Info2"
    phenotype_entries = row_string.split(', ')
    mapped_entries = [] # Lista para guardar las nuevas cadenas "Cat:Info (ID: ...)"

    for entry in phenotype_entries:
        try:
            # Separa "Categoría:Información"
            cat, inf = entry.split(':', 1)
            cat = cat.strip()
            inf = inf.strip()
            inf_lower = inf.lower() # Usar minúsculas para la búsqueda

            found_id = "ID_NA" # Valor por defecto

            # --- Lógica de Mapeo por Categoría ---
            
            
            match_tuple = process.extractOne(inf_lower, side_effect_choices, scorer=fuzz.token_sort_ratio, score_cutoff=70)
                    
            if match_tuple:
                        # match_tuple es (match_string, score)
                    best_match = match_tuple[0]
                    # Obtenemos el ID usando el string coincidente (que es una clave en nuestro map)
                    found_id = side_effect_map[best_match]

                    mapped_entries.append(f"{cat}:{found_id}")
            else:
                mapped_entries.append(f"{cat}:{inf}")
                j+=1
                list_indexes_not_mapped.append((idx, entry))
                
                
                        
                        
                
            
            # --- Fin de la Lógica de Mapeo ---

            # Añadir la entrada formateada a nuestra lista
            

        except ValueError:
            # Si la entrada no tiene el formato "Cat:Info" (p.ej. está vacía), se añade tal cual
            mapped_entries.append(entry)
    
    # Unir todas las entradas mapeadas ("Cat:Info (ID:...)") con comas
    # y asignarlas a la nueva columna en la fila correspondiente (idx)
    df.loc[idx, 'Effect_phenotype_id'] = ', '.join(mapped_entries) #type: ignore

########    SEGUNDO MAPEO PARA LA OTRA COLUMNA   ##########
j=0
list_indexes_not_mapped = []
mask = df['Pop_Phenotypes/Diseases'].notna()


# 2. Crear una nueva columna para guardar los resultados del mapeo
df['Pop_phenotype_id'] = pd.NA

# 3. Iterar sobre el dataframe (en las filas limpiadas) para mapear IDs
#    Usamos la misma 'mask' que antes para operar solo sobre las filas relevantes.
for idx, row_string in df.loc[mask, 'Pop_Phenotypes/Diseases'].items():

    if not isinstance(row_string, str) or not row_string:
        continue

    # La fila puede tener múltiples entradas: "Cat1:Info1, Cat2:Info2"
    phenotype_entries = row_string.split(', ')
    mapped_entries = [] # Lista para guardar las nuevas cadenas "Cat:Info (ID: ...)"

    for entry in phenotype_entries:
        try:
            # Separa "Categoría:Información"
            cat, inf = entry.split(':', 1)
            cat = cat.strip()
            inf = inf.strip()
            inf_lower = inf.lower() # Usar minúsculas para la búsqueda

            found_id = "ID_NA" # Valor por defecto

            # --- Lógica de Mapeo por Categoría ---
            
            
            match_tuple = process.extractOne(inf_lower, side_effect_choices, scorer=fuzz.token_sort_ratio, score_cutoff=70)
                    
            if match_tuple:
                        # match_tuple es (match_string, score)
                    best_match = match_tuple[0]
                    # Obtenemos el ID usando el string coincidente (que es una clave en nuestro map)
                    found_id = side_effect_map[best_match]

                    mapped_entries.append(f"{cat}:{found_id}")
            else:
                mapped_entries.append(f"{cat}:{inf}")
                j+=1
                list_indexes_not_mapped.append((idx, entry))
                
                
                        
                        
                
            
            # --- Fin de la Lógica de Mapeo ---

            # Añadir la entrada formateada a nuestra lista
            

        except ValueError:
            # Si la entrada no tiene el formato "Cat:Info" (p.ej. está vacía), se añade tal cual
            mapped_entries.append(entry)
    
    # Unir todas las entradas mapeadas ("Cat:Info (ID:...)") con comas
    # y asignarlas a la nueva columna en la fila correspondiente (idx)
    df.loc[idx, 'Pop_phenotype_id'] = ', '.join(mapped_entries) #type: ignore
    
########    FIN SEGUNDO MAPEO   ##########


df = df.to_csv('final_test.tsv', sep='\t', index=False)


#print(f"Número de entradas no mapeadas: {j}")

#print("Mapeo de ID completado.")




# Opcional: Mostrar las primeras filas del resultado para verificar
# print("\nResultados del mapeo:")
# pd.set_option('display.max_colwidth', None) # Para ver el contenido completo
# print(df.loc[mask, ['Effect_phenotype', 'Effect_phenotype_Mapped_ID']].head())

# --- FIN DEL CÓDIGO DE MAPEO ---

'''n = 0

for idx, field in df[['Effect_phenotype', 'Effect_phenotype_id']].iterrows():
    match = re.match(r'.*ID_No_Encontrado.*', str(field['Effect_phenotype_id']))
    if match:
        n+=1
        print(idx, field)

print(f"Número de entradas con 'ID_No_Encontrado': {n}")
print(len(df))

print("porcentaje = ", (n/len(df))*100)
'''