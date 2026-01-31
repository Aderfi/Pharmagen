# Pharmagen Environment Setup for Windows

param (
    [switch]$Clean = $false
)

# Colores
$Green = "Green"
$Blue = "Cyan"
$Red = "Red"
$Yellow = "Yellow"

Write-Host "========================================" -ForegroundColor $Blue
Write-Host "   💊 Pharmagen Environment Setup       " -ForegroundColor $Blue
Write-Host "========================================" -ForegroundColor $Blue

# 1. Comprobar si uv está instalado
if (-not (Get-Command "uv" -ErrorAction SilentlyContinue)) {
    Write-Host "⚠️  uv no está instalado." -ForegroundColor $Yellow
    $choice = Read-Host "¿Quieres instalar uv ahora? (s/n)"
    if ($choice -match "^[Ss]") {
        Write-Host "⬇️  Instalando uv..." -ForegroundColor $Blue
        powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
        # Refrescar variables de entorno para la sesión actual es complicado en PS sin reiniciar,
        # así que avisamos.
        Write-Host "✅ uv instalado. Por favor, cierra y vuelve a abrir PowerShell para usarlo." -ForegroundColor $Green
        exit
    } else {
        Write-Host "❌ Error: Necesitas uv para continuar." -ForegroundColor $Red
        exit 1
    }
} else {
    Write-Host "✅ uv detectado." -ForegroundColor $Green
}

# 2. Opción de limpieza
if ($Clean) {
    Write-Host "🧹 Limpiando entorno antiguo (.venv y uv.lock)..." -ForegroundColor $Yellow
    if (Test-Path ".venv") { Remove-Item -Recurse -Force ".venv" }
    if (Test-Path "uv.lock") { Remove-Item -Force "uv.lock" }
    Write-Host "✅ Limpieza completada." -ForegroundColor $Green
}

# 3. Sincronización del entorno
Write-Host "🚀 Creando entorno virtual e instalando dependencias..." -ForegroundColor $Blue
Write-Host "ℹ️  Instalando grupos: default, train, dev" -ForegroundColor $Blue
Write-Host "ℹ️  Usando índice CUDA: cu124 (PyTorch)" -ForegroundColor $Blue

# Sincroniza e instala TODAS las dependencias
# uv gestiona automáticamente las diferencias de plataforma (ignora pysam en Windows por el marcador que pusimos)
uv sync --all-extras

if ($LASTEXITCODE -eq 0) {
    Write-Host "========================================" -ForegroundColor $Blue
    Write-Host "✅ ¡Instalación completada con éxito!" -ForegroundColor $Green
    Write-Host "========================================" -ForegroundColor $Blue
    Write-Host "Para activar el entorno:"
    Write-Host "   .venv\Scripts\activate" -ForegroundColor $Yellow
    Write-Host ""
    Write-Host "Para ejecutar el programa:"
    Write-Host "   uv run python main.py --mode menu" -ForegroundColor $Yellow
} else {
    Write-Host "❌ Hubo un error durante la instalación." -ForegroundColor $Red
    exit 1
}
