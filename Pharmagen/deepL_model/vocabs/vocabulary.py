"""
Uso de la clasificación ATC como diccionario para agrupar medicamentos.
El sistema ATC tiene 5 niveles de jerarquía



"""

ATC_dictionary ={
    "J05": {  # Antivirales de uso sistémico
        "J05A": {
            "J05AF": ["J05AF01", "J05AF02"],  # Ejemplo: NRTI
            "J05AG": ["J05AG01", "J05AG02"],  # Ejemplo: NNRTI
        }
    },
    "C07": {  # Betabloqueantes
        "C07A": {
            "C07AB": ["C07AB02", "C07AB03"]  # Ejemplo: metoprolol, bisoprolol
        }
    }
}