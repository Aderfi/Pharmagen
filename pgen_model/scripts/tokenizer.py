import json
import numpy as np
from keras.layers import TextVectorization
from keras.models import Model
from keras.layers import Input, Embedding, Flatten, Dense, Concatenate

# 1. Carga tus datos (ejemplo para farmacos)
with open("Pharmagen/deepL_model/vocabs/farmacos.json", "r", encoding="utf8") as f:
    farmacos = json.load(f)

# Ejemplo: lista de códigos ATC y lista de polimorfismos
lista_codigos = [f["codigo"] for f in farmacos]  # Todos los códigos ATC posibles
lista_poli = ["CYP2C19*1", "CYP2C19*2", "SLCO1B1*5", "SLCO1B1*1"]  # Ejemplo de polimorfismos posibles

# 2. Define el vectorizador para códigos ATC
max_codigos = 5  # Número máximo de medicamentos por paciente
vectorizador_atc = TextVectorization(
    output_mode="int",
    output_sequence_length=max_codigos,
)
vectorizador_atc.adapt(np.array(lista_codigos))

# 3. Define el vectorizador para polimorfismos
max_poli = 20  # Número máximo de polimorfismos por paciente
vectorizador_poli = TextVectorization(
    output_mode="int",
    output_sequence_length=max_poli,
)
vectorizador_poli.adapt(np.array(lista_poli))

# 4. Ejemplo de paciente
codigos_paciente = ["A01AB03", "B01AA03"]
polimorfismos_paciente = ["CYP2C19*1", "SLCO1B1*5"]

# Convierte a tensores de secuencia de índices
codigos_vec = vectorizador_atc(np.array([codigos_paciente]))
poli_vec = vectorizador_poli(np.array([polimorfismos_paciente]))

print("Secuencia códigos ATC:", codigos_vec.numpy())
print("Secuencia polimorfismos:", poli_vec.numpy())

# 5. Ejemplo de modelo Keras
med_input = Input(shape=(max_codigos,), name='medicamentos')
geno_input = Input(shape=(max_poli,), name='polimorfismos')

med_emb = Embedding(input_dim=len(vectorizador_atc.get_vocabulary()), output_dim=32, mask_zero=True)(med_input)
geno_emb = Embedding(input_dim=len(vectorizador_poli.get_vocabulary()), output_dim=16, mask_zero=True)(geno_input)

med_flat = Flatten()(med_emb)
geno_flat = Flatten()(geno_emb)

x = Concatenate()([med_flat, geno_flat])
x = Dense(64, activation='relu')(x)
x = Dense(32, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=[med_input, geno_input], outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
