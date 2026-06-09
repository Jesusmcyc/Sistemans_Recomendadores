# setup.ps1 — Instalar y verificar dependencias del proyecto

$PYTHON = ".\.venv\Scripts\python.exe"
$PIP    = ".\.venv\Scripts\pip.exe"

# ── 1. Verificar que el entorno virtual existe ─────────────────────────────────
if (-not (Test-Path $PYTHON)) {
    Write-Host "Creando entorno virtual..." -ForegroundColor Cyan
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: No se pudo crear el entorno virtual. Asegurate de tener Python instalado." -ForegroundColor Red
        exit 1
    }
}

# ── 2. Instalar dependencias ───────────────────────────────────────────────────
Write-Host ""
Write-Host "Instalando dependencias desde requirements.txt..." -ForegroundColor Cyan
& $PIP install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org -r requirements.txt --quiet

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Fallo la instalacion de dependencias." -ForegroundColor Red
    exit 1
}

# ── 3. Verificar imports ───────────────────────────────────────────────────────
Write-Host ""
Write-Host "Verificando instalacion..." -ForegroundColor Cyan

$verificacion = @"
import sys

paquetes = {
    'pandas':      'pandas',
    'numpy':       'numpy',
    'sklearn':     'scikit-learn',
    'matplotlib':  'matplotlib',
    'seaborn':     'seaborn',
    'nbformat':    'nbformat',
    'jupyter':     'jupyter_client',
}

ok = True
for modulo, nombre in paquetes.items():
    try:
        m = __import__(modulo)
        version = getattr(m, '__version__', 'ok')
        print(f'  OK  {nombre:<20} {version}')
    except ImportError:
        print(f'  FALLO  {nombre}')
        ok = False

if ok:
    print()
    print('Todas las dependencias instaladas correctamente.')
    sys.exit(0)
else:
    print()
    print('Algunas dependencias no se instalaron. Revisa los errores anteriores.')
    sys.exit(1)
"@

& $PYTHON -c $verificacion

if ($LASTEXITCODE -ne 0) {
    exit 1
}

# ── 4. Verificar datos ─────────────────────────────────────────────────────────
Write-Host ""
Write-Host "Verificando archivos de datos..." -ForegroundColor Cyan

$archivos = @(
    @{ path = "data\movies_metadata.csv"; requerido = $true  },
    @{ path = "data\keywords.csv";        requerido = $true  },
    @{ path = "data\credits.csv";         requerido = $false }
)

foreach ($a in $archivos) {
    if (Test-Path $a.path) {
        $mb = [math]::Round((Get-Item $a.path).Length / 1MB, 1)
        Write-Host "  OK  $($a.path) ($mb MB)"
    } elseif ($a.requerido) {
        Write-Host "  FALTA  $($a.path)  (requerido)" -ForegroundColor Red
    } else {
        Write-Host "  AVISO  $($a.path) no encontrado — necesario para la Seccion 4" -ForegroundColor Yellow
        Write-Host "         Descargarlo de: https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset"
    }
}

Write-Host ""
Write-Host "Listo. Ejecuta el notebook con:" -ForegroundColor Green
Write-Host "  .\.venv\Scripts\jupyter.exe notebook recommender_system.ipynb"
