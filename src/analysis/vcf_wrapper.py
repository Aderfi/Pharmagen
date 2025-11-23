import subprocess
from pathlib import Path


def ejecutar_analisis_cpp(vcf_path: Path, fasta_path: Path):
    binario = Path("src/bin/vcf_tool") # Ruta a tu ejecutable compilado
    
    if not binario.exists():
        raise FileNotFoundError("Debes compilar el módulo C++ primero.")

    print("🚀 Iniciando motor C++ de alto rendimiento...")
    
    process = subprocess.Popen(
        [str(binario), str(vcf_path), str(fasta_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # Leer salida en tiempo real (Streaming)
    while True:
        line = process.stdout.readline() if process.stdout else ''
        if not line and process.poll() is not None:
            break
        if line:
            print(line.strip()) # O procesar/guardar en base de datos

    if process.returncode != 0:
        print(f"Error en C++: {process.stderr.read() if process.stderr else 'No se obtuvo mensaje de error.'}")