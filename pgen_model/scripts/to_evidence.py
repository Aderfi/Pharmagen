""""

import pandas as pd
evidence = pd.read_csv("conocimiento_farmacogenetico.csv")

def check_riesgo(farmaco, gen, alelo):
    match = evidence[
        (evidence['farmaco'] == farmaco) &
        (evidence['gen'] == gen) &
        (evidence['alelo'] == alelo)
    ]
    if match.empty:
        return None
    else:
        return match[['riesgo', 'tipo_riesgo']].to_dict('records')

# Output: [{'riesgo': 'fracaso_terapeutico', 'tipo_riesgo': 'metabolizador_lento'}]
print(check_riesgo("clopidogrel", "CYP2C19", "*2"))

Copilot said: ¡Perfecto! Entonces tu **archivo de conocimiento**

¡Perfecto!
Entonces tu archivo de conocimiento puede ser aún más simple, por ejemplo en .csv:
gen	alelo	farmaco	riesgo	tipo_riesgo
CYP2C19	*2	clopidogrel	fracaso_terapeutico	metabolizador_lento
SLCO1B1	*5	simvastatina	efectos_adversos	miopatía
...	...	...	...	...

No necesitas el campo "evidencia" porque, como dices, ya has filtrado las asociaciones más robustas.

"""