MODEL_CONFIGS = {
    
    #['Drug', 'Gene','Allele', 'Genotype', 'Outcome', 'Variation', 'Effect']
    #
    #"Outcome-Variation-Effect-Entity": {
    #    "targets": ["Outcome", "Variation", "Effect", "Entity"],
    #    "cols": ['Drug', 'Gene','Allele', 'Genotype', 'Outcome', 'Variation', 'Effect']
    #},
    "Outcome-Variation-Effect": {
        "targets": ["Outcome", "Variation", "Effect"],
        "cols": ['Drug', 'Gene','Allele', 'Genotype', 'Outcome', 'Variation', 'Effect']
    },
    "Effect-Entity": {
        "targets": ["Effect", "Entity"],
        "cols": ['Drug', 'Gene','Allele', 'Genotype', 'Effect', 'Entity']
    },
    "Variation-Effect": {
        "targets": ["Variation", "Effect"],
        "cols": ['Drug', 'Gene','Allele', 'Genotype', 'Variation', 'Effect']
    },
    "Variation-Effect-Entity": {
        "targets": ["Variation", "Effect", "Entity"],
        "cols": ['Drug', 'Gene','Allele', 'Genotype', 'Variation', 'Effect', 'Entity']
    }
}