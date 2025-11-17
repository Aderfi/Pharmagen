import pandas as pd

FILE_PATH = 'train_data/relationships_associated_corrected_mapped.tsv'

df = pd.read_csv(FILE_PATH, sep='\t')

print(f"\nCargando datos desde {FILE_PATH}...")

mask = (df['Entity1_type'] == 'Chemical') | (df['Entity1_type'] == 'Gene')
mask2 = (df['Entity2_type'] == 'Chemical') | (df['Entity2_type'] == 'Gene')

filt_df = df[mask & mask2]

order1_df = (filt_df['Entity1_type'] == 'Chemical') & (filt_df['Entity2_type'] == 'Gene')
order2_df = (filt_df['Entity2_type'] == 'Chemical') & (filt_df['Entity1_type'] == 'Gene')


merged_df = pd.DataFrame(columns=['Drug', 'Gene'])
merged_df = pd.concat([filt_df[order1_df][['Entity1_name', 'Entity2_name']].rename(columns={'Entity1_name': 'Drug', 'Entity2_name': 'Gene'}),
                           filt_df[order2_df][['Entity2_name', 'Entity1_name']].rename(columns={'Entity2_name': 'Drug', 'Entity1_name': 'Gene'})],
                          ignore_index=True)

merged_df = merged_df.drop_duplicates().reset_index(drop=True)

print(merged_df.shape)

merged_df.to_csv('train_data/drug_gene_edges.tsv', sep='\t', index=False)