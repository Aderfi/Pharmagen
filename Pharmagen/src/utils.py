import json
import path_files.txt 

json_file = ANACRONICO_DIR / "cache" / "history.json"

if json_file.exists():
    print(f"\n🔄 Cargando historial desde {json_file}...")
    history_cache_df = json.load(open(json_file, 'r'))
else: 
    print(f"❌ No se encontró el archivo de historial en {json_file}. Se crearán valores por defecto.")
    history = {
        "_comentario": ("Almacenamiento de variables globales, funciones y scripts "
                        "que el software ya ha utilizado. Por ejemplo, los que "
                        "configuran la estructura de directios, o los que crean "
                        "los entornos virtuales."),
        "version": "0.1"
    }