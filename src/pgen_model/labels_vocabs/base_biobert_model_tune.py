import pandas as pd
import torch
from datasets import Dataset, ClassLabel
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df1 = pd.read_csv(
    "var_drug_ann_model2.csv",
    sep=";",
    usecols=["Drug", "Genotype", "Outcome", "Variation"],
)
df2 = pd.read_csv(
    "var_fa_ann_model.csv",
    sep=";",
    usecols=["Drug", "Genotype", "Outcome", "Variation"],
)
df3 = pd.read_csv(
    "var_pheno_ann_model.csv",
    sep=";",
    usecols=["Drug", "Genotype", "Outcome", "Variation"],
)

genes_df = (
    pd.concat([df1["Genotype"], df2["Genotype"], df3["Genotype"]])
    .dropna()
    .unique()
    .tolist()
)


df = pd.concat(
    [df1, df2, df3],
    ignore_index=True,
)  # columnas: text1, text2

df["Outcome"] = df["Outcome"].fillna("")
df["Variation"] = df["Variation"].fillna("")


def label_por_gen(row):
    genes = genes_df  # usa la lista de genes generada
    for gen in genes:
        if gen in row["Outcome"] and gen in row["Variation"]:
            return "relacionado"
    return "no_relacionado"


df["label"] = df.apply(label_por_gen, axis=1)
# df.to_csv("pares_con_labels.csv", index=False)


# Paso 1: Cargar y preparar los datos
# df = pd.read_csv("datos_entrenamiento.csv")  # Asegúrate de tener las columnas: text1, text2, label
# df = df.dropna(subset=['text1', 'text2', 'label'])

# Convertir las etiquetas a números
classes = sorted(df["label"].unique())
class2id = {c: i for i, c in enumerate(classes)}
df["label"] = df["label"].map(class2id)

# Concatenar ambos textos con [SEP]
df["input"] = df["Drug"].astype(str) + " [SEP] " + df["Genotype"].astype(str)

# Crear HuggingFace Dataset
dataset = Dataset.from_pandas(df[["input", "label"]])
dataset = dataset.train_test_split(test_size=0.1, seed=42)

# Paso 2: Tokenización
model_name = "dmis-lab/biobert-base-cased-v1.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)


def preprocess(example):
    return tokenizer(
        example["input"], truncation=True, padding="max_length", max_length=128
    )


tokenized = dataset.map(preprocess, batched=True)

# Paso 3: Cargar modelo para clasificación
num_labels = len(classes)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=num_labels
)

# Paso 4: Entrenamiento
training_args = TrainingArguments(
    output_dir="./biobert-pares-fine-tuned",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=2,
)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = np.mean(preds == labels)
    return {"accuracy": acc}


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model("./biobert-pares-fine-tuned")
tokenizer.save_pretrained("./biobert-pares-fine-tuned")
print("Fine-tuning terminado. Modelo guardado en ./biobert-pares-fine-tuned")
print("Clases:", class2id)
