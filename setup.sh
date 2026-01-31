#!/bin/bash

# Colores para la salida
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}   💊 Pharmagen Environment Setup      ${NC}"
echo -e "${BLUE}========================================${NC}"

# 1. Comprobar si uv está instalado
if ! command -v uv &> /dev/null; then
    echo -e "${YELLOW}⚠️  uv no está instalado.${NC}"
    read -p "¿Quieres instalar uv ahora? (s/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Ss]$ ]]; then
        echo -e "${BLUE}⬇️  Instalando uv...${NC}"
        curl -LsSf https://astral.sh/uv/install.sh | sh
        # Intentar cargar el entorno si se acaba de instalar
        source "$HOME/.cargo/env" 2>/dev/null || true
    else
        echo -e "${RED}❌ Error: Necesitas uv para continuar.${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}✅ uv detectado: $(uv --version)${NC}"
fi

# 2. Opción de limpieza
if [[ "$1" == "--clean" ]]; then
    echo -e "${YELLOW}🧹 Limpiando entorno antiguo (.venv y uv.lock)...${NC}"
    rm -rf .venv uv.lock
    echo -e "${GREEN}✅ Limpieza completada.${NC}"
fi

# 3. Sincronización del entorno
echo -e "${BLUE}🚀 Creando entorno virtual e instalando dependencias...${NC}"
echo -e "${BLUE}ℹ️  Instalando grupos: default, train, dev${NC}"
echo -e "${BLUE}ℹ️  Usando índice CUDA: cu130 (PyTorch)${NC}"

# Sincroniza e instala TODAS las dependencias (dev y train incluidas)
# Si solo quisieras producción: uv sync --no-dev --no-group train
uv sync --all-extras

if [ $? -eq 0 ]; then
    echo -e "${BLUE}========================================${NC}"
    echo -e "${GREEN}✅ ¡Instalación completada con éxito!${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo -e "Para activar el entorno:"
    echo -e "   ${YELLOW}source .venv/bin/activate${NC}"
    echo -e ""
    echo -e "Para ejecutar el programa:"
    echo -e "   ${YELLOW}uv run python main.py --mode menu${NC}"
else
    echo -e "${RED}❌ Hubo un error durante la instalación.${NC}"
    exit 1
fi
