MODEL_CONFIGS = {
    "Outcome-Variation-Effect-Entity": {
        "targets": ["Outcome", "Variation", "Effect", "Entity"],
        "cols": ["Drug", "Genotype", "Outcome", "Variation", "Effect", "Entity"]
    },
    "Outcome-Variation": {
        "targets": ["Outcome", "Variation"],
        "cols": ["Drug", "Genotype", "Outcome", "Variation"]
    },
    "Effect-Entity": {
        "targets": ["Effect", "Entity"],
        "cols": ["Drug", "Genotype", "Effect", "Entity"]
    },
    "Variation-Effect": {
        "targets": ["Variation", "Effect"],
        "cols": ["Drug", "Genotype", "Variation", "Effect"]
    }
}