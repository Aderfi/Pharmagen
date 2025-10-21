MODEL_CONFIGS = {
    
    #['Drug', 'Gene','Allele', 'Genotype', 'Outcome', 'Variation', 'Effect']
    #
    #"Outcome-Variation-Effect-Entity": {
    #    "targets": ["Outcome", "Variation", "Effect", "Entity"],
    #    "cols": ['Drug', 'Gene','Allele', 'Genotype', 'Outcome', 'Variation', 'Effect']
    #},

    # ["Drug", "Gene", "Allele", "Genotype", "Outcome_category", "Effect_direction", "Effect_category", "Entity", "Entity_name", "Affected_Pop", "Therapeutic_Outcome"]

    "Outcome-Effect-Entity --> Therapeutic_Outcome": {
        "targets": ["Outcome", "Effect_direction", "Effect_category", "Entity", "Entity_name", "Therapeutic_Outcome"],
        "cols": ["Drug", "Gene", "Allele", "Genotype", "Outcome_category", "Effect_direction", "Effect_category", "Entity", "Entity_name", "Affected_Pop", "Therapeutic_Outcome"]

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