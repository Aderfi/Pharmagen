import subprocess
import sys
import venv
import shutil

# Definir la versión de Python y el nombre del entorno
PYTHON_VERSION = "3.13.7"
ENV_NAME = "EnvMaster"
REQUIREMENTS_FILE = "requirements.txt"


def check_python_version(version):
    """Verifica si una versión específica de Python está instalada y disponible."""
    try:
        # Usa 'py' en Windows y 'python'/'pythonX.Y' en otros sistemas
        command = [f"python{version}", "--version"]
        if sys.platform == "win32":
            command = [f"py-{version}", "--version"]

        subprocess.run(
            command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        print(f"✅ La versión de Python {version} está disponible.")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(
            f"❌ La versión de Python {version} no se encuentra. Por favor, instálala."
        )
        return False


def create_virtual_environment(env_name, python_version):
    """Crea un entorno virtual con la versión de Python especificada."""
    if shutil.which(env_name):
        print(f"⚠️ El entorno virtual '{env_name}' ya existe. Eliminándolo...")
        shutil.rmtree(env_name)

    print(f"🔧 Creando el entorno virtual '{env_name}' con Python {python_version}...")
    try:
        # Determina el ejecutable de Python adecuado
        if sys.platform == "win32":
            python_executable = shutil.which(f"py -{python_version}") or shutil.which(
                f"python{python_version}"
            )
        else:
            python_executable = shutil.which(f"python{python_version}")

        if not python_executable:
            raise FileNotFoundError(
                f"No se encontró el ejecutable de Python {python_version}."
            )

        # Crea el entorno virtual usando el ejecutable encontrado
        subprocess.run(
            [python_executable, "-m", "venv", env_name, "--clear", "--upgrade-deps"],
            check=True,
        )
        print(f"✅ Entorno virtual '{env_name}' creado exitosamente.")
        return True
    except Exception as e:
        print(f"❌ Falló la creación del entorno virtual: {e}")
        return False


def install_requirements(env_name, requirements_file):
    """Instala las dependencias del archivo de requerimientos en el entorno virtual."""
    print(f"📦 Instalando dependencias desde '{requirements_file}'...")
    try:
        # Define el path del ejecutable de pip dentro del entorno virtual
        pip_path = (
            f"{env_name}/bin/pip"
            if sys.platform != "win32"
            else f"{env_name}/Scripts/pip.exe"
        )

        subprocess.run([pip_path, "install", "-r", requirements_file], check=True)
        print("✅ Todas las dependencias se instalaron correctamente.")
        return True
    except FileNotFoundError:
        print(
            f"❌ No se pudo encontrar el archivo de requerimientos: {requirements_file}"
        )
        return False
    except subprocess.CalledProcessError as e:
        print(f"❌ Falló la instalación de las dependencias. Error: {e}")
        return False


def main():
    """Función principal para ejecutar todo el proceso."""
    if not check_python_version(PYTHON_VERSION):
        return

    if not create_virtual_environment(ENV_NAME, PYTHON_VERSION):
        return

    if not install_requirements(ENV_NAME, REQUIREMENTS_FILE):
        return

    print("\n🎉 ¡El proceso ha finalizado! Para activar el entorno, usa:")
    if sys.platform == "win32":
        print(f"  .\\{ENV_NAME}\\Scripts\\activate")
    else:
        print(f"  source ./{ENV_NAME}/bin/activate")


if __name__ == "__main__":
    main()
