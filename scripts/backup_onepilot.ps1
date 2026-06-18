# ═══════════════════════════════════════════════════════════════
# OnePilot v0 — Backup Script
# Sauvegarde complète : DB, modèles, notebooks, config
# Usage : .\scripts\backup_onepilot.ps1
# ═══════════════════════════════════════════════════════════════

# Détecte automatiquement le dossier projet (peu importe son nom)
$projectDir = Split-Path -Parent $PSScriptRoot
$backupRoot = "C:\Users\user\Desktop\onepilot_backup"
$timestamp  = Get-Date -Format "yyyy-MM-dd_HH-mm"
$backupDir  = "$backupRoot\$timestamp"

Write-Host ""
Write-Host "══════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "   OnePilot — Backup Complet              " -ForegroundColor Cyan
Write-Host "   $timestamp                             " -ForegroundColor Cyan
Write-Host "══════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

# ── Créer le dossier de backup ────────────────────────────────
New-Item -ItemType Directory -Force -Path $backupDir | Out-Null
Write-Host "[ 1/6 ] Dossier backup : $backupDir" -ForegroundColor Yellow

# ── 2. Dump PostgreSQL ────────────────────────────────────────
Write-Host "[ 2/6 ] Dump PostgreSQL..." -ForegroundColor Yellow
$pgStatus = docker inspect --format="{{.State.Running}}" onepilot_postgres 2>$null
if ($pgStatus -eq "true") {
    docker exec onepilot_postgres pg_dump -U onepilot onepilot_dev > "$backupDir\onepilot_db.sql"
    if (Test-Path "$backupDir\onepilot_db.sql") {
        $size = (Get-Item "$backupDir\onepilot_db.sql").Length / 1KB
        Write-Host "  ✓ DB dumpée : $([math]::Round($size, 1)) KB" -ForegroundColor Green
    } else {
        Write-Host "  ✗ Dump échoué" -ForegroundColor Red
    }
} else {
    Write-Host "  ✗ Container onepilot_postgres non démarré" -ForegroundColor Red
}

# ── 3. Modèles ML (.pkl) ──────────────────────────────────────
Write-Host "[ 3/6 ] Backup modèles ML..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path "$backupDir\models" | Out-Null
$pklFiles = Get-ChildItem -Path "$projectDir\api" -Filter "*.pkl" -ErrorAction SilentlyContinue
if ($pklFiles) {
    foreach ($f in $pklFiles) {
        Copy-Item $f.FullName "$backupDir\models\" -Force
        Write-Host "  ✓ $($f.Name)" -ForegroundColor Green
    }
    $pklModels = Get-ChildItem -Path "$projectDir\api\models" -Filter "*.pkl" -ErrorAction SilentlyContinue
    if ($pklModels) {
        New-Item -ItemType Directory -Force -Path "$backupDir\models\models" | Out-Null
        foreach ($f in $pklModels) {
            Copy-Item $f.FullName "$backupDir\models\models\" -Force
            Write-Host "  ✓ models\$($f.Name)" -ForegroundColor Green
        }
    }
} else {
    Write-Host "  ⚠ Aucun .pkl trouvé" -ForegroundColor Yellow
}

# ── 4. Notebooks ──────────────────────────────────────────────
Write-Host "[ 4/6 ] Backup notebooks..." -ForegroundColor Yellow
if (Test-Path "$projectDir\notebooks") {
    New-Item -ItemType Directory -Force -Path "$backupDir\notebooks" | Out-Null
    $nbFiles = Get-ChildItem -Path "$projectDir\notebooks" -Filter "*.ipynb" -ErrorAction SilentlyContinue
    foreach ($f in $nbFiles) {
        Copy-Item $f.FullName "$backupDir\notebooks\" -Force
    }
    Write-Host "  ✓ $($nbFiles.Count) notebooks sauvegardés" -ForegroundColor Green
} else {
    Write-Host "  ✗ Dossier notebooks introuvable" -ForegroundColor Red
}

# ── 5. Fichiers de config ─────────────────────────────────────
Write-Host "[ 5/6 ] Backup config..." -ForegroundColor Yellow
$configFiles = @(".env", "docker-compose.yml", "Dockerfile", "pyproject.toml")
New-Item -ItemType Directory -Force -Path "$backupDir\config" | Out-Null
foreach ($f in $configFiles) {
    $src = "$projectDir\$f"
    if (Test-Path $src) {
        Copy-Item $src "$backupDir\config\" -Force
        Write-Host "  ✓ $f" -ForegroundColor Green
    }
}

# ── 6. FastText modèles entraînés ────────────────────────────
Write-Host "[ 6/6 ] Backup modèles FastText..." -ForegroundColor Yellow
$binFiles = Get-ChildItem -Path $projectDir -Filter "*.bin" -Recurse -ErrorAction SilentlyContinue
if ($binFiles) {
    New-Item -ItemType Directory -Force -Path "$backupDir\fasttext" | Out-Null
    foreach ($f in $binFiles) {
        Copy-Item $f.FullName "$backupDir\fasttext\" -Force
        Write-Host "  ✓ $($f.Name)" -ForegroundColor Green
    }
} else {
    Write-Host "  ⚠ Aucun modèle FastText (.bin) trouvé" -ForegroundColor Yellow
}

# ── Résumé ────────────────────────────────────────────────────
Write-Host ""
Write-Host "══════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "   Backup terminé !                       " -ForegroundColor Cyan
Write-Host "══════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "   Dossier : $backupDir" -ForegroundColor White
$totalSize = (Get-ChildItem -Path $backupDir -Recurse | Measure-Object -Property Length -Sum).Sum / 1MB
Write-Host "   Taille  : $([math]::Round($totalSize, 1)) MB" -ForegroundColor White
Write-Host ""
