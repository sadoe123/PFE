# ═══════════════════════════════════════════════════════════════
# OnePilot v0 — Restore Script
# Restaure complètement le projet depuis un backup
# Usage : .\scripts\restore_onepilot.ps1 -BackupDir "C:\Users\user\Desktop\onepilot_backup\2026-06-13_01-30"
# ═══════════════════════════════════════════════════════════════

param(
    [Parameter(Mandatory=$false)]
    [string]$BackupDir = ""
)

$projectDir = Split-Path -Parent $PSScriptRoot

Write-Host ""
Write-Host "══════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "   OnePilot — Restauration Complète       " -ForegroundColor Cyan
Write-Host "══════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

# ── Trouver le dernier backup si non spécifié ─────────────────
if (-not $BackupDir) {
    $backupRoot = "C:\Users\user\Desktop\onepilot_backup"
    if (Test-Path $backupRoot) {
        $lastBackup = Get-ChildItem -Path $backupRoot -Directory | Sort-Object Name -Descending | Select-Object -First 1
        if ($lastBackup) {
            $BackupDir = $lastBackup.FullName
            Write-Host "  ℹ️  Dernier backup trouvé : $BackupDir" -ForegroundColor Cyan
        } else {
            Write-Host "  ✗ Aucun backup trouvé dans $backupRoot" -ForegroundColor Red
            exit 1
        }
    } else {
        Write-Host "  ✗ Dossier backup introuvable : $backupRoot" -ForegroundColor Red
        exit 1
    }
}

if (-not (Test-Path $BackupDir)) {
    Write-Host "  ✗ Dossier backup introuvable : $BackupDir" -ForegroundColor Red
    exit 1
}

Write-Host "  Backup : $BackupDir" -ForegroundColor White
Write-Host ""

# ── 1. Vérifier que Docker tourne ────────────────────────────
Write-Host "[ 1/6 ] Vérification Docker..." -ForegroundColor Yellow
$pgStatus = docker inspect --format="{{.State.Running}}" onepilot_postgres 2>$null
if ($pgStatus -ne "true") {
    Write-Host "  ⚠ Containers non démarrés — lancement..." -ForegroundColor Yellow
    Set-Location $projectDir
    docker-compose up -d
    Start-Sleep -Seconds 15
    $pgStatus = docker inspect --format="{{.State.Running}}" onepilot_postgres 2>$null
    if ($pgStatus -ne "true") {
        Write-Host "  ✗ Impossible de démarrer les containers" -ForegroundColor Red
        exit 1
    }
}
Write-Host "  ✓ Containers actifs" -ForegroundColor Green

# ── 2. Restaurer PostgreSQL ───────────────────────────────────
Write-Host "[ 2/6 ] Restauration PostgreSQL..." -ForegroundColor Yellow
$dbFile = "$BackupDir\onepilot_db.sql"
if (Test-Path $dbFile) {
    # Copier le dump dans le container
    docker cp $dbFile onepilot_postgres:/tmp/onepilot_db.sql
    # Vider la DB et restaurer
    docker exec onepilot_postgres psql -U onepilot -d onepilot_dev -c "DROP SCHEMA public CASCADE; CREATE SCHEMA public;" 2>$null
    docker exec onepilot_postgres psql -U onepilot -d onepilot_dev -f /tmp/onepilot_db.sql 2>$null
    Write-Host "  ✓ Base PostgreSQL restaurée" -ForegroundColor Green
} else {
    Write-Host "  ✗ Fichier dump introuvable : $dbFile" -ForegroundColor Red
}

# ── 3. Restaurer modèles ML (.pkl) ───────────────────────────
Write-Host "[ 3/6 ] Restauration modèles ML..." -ForegroundColor Yellow
$modelsDir = "$BackupDir\models"
if (Test-Path $modelsDir) {
    $pklFiles = Get-ChildItem -Path $modelsDir -Filter "*.pkl" -ErrorAction SilentlyContinue
    foreach ($f in $pklFiles) {
        Copy-Item $f.FullName "$projectDir\api\" -Force
        Write-Host "  ✓ $($f.Name)" -ForegroundColor Green
    }
    # models/ subfolder
    $pklModels = Get-ChildItem -Path "$modelsDir\models" -Filter "*.pkl" -ErrorAction SilentlyContinue
    if ($pklModels) {
        New-Item -ItemType Directory -Force -Path "$projectDir\api\models" | Out-Null
        foreach ($f in $pklModels) {
            Copy-Item $f.FullName "$projectDir\api\models\" -Force
            Write-Host "  ✓ models\$($f.Name)" -ForegroundColor Green
        }
    }
} else {
    Write-Host "  ⚠ Dossier modèles introuvable" -ForegroundColor Yellow
}

# ── 4. Restaurer FastText ─────────────────────────────────────
Write-Host "[ 4/6 ] Restauration modèles FastText..." -ForegroundColor Yellow
$ftDir = "$BackupDir\fasttext"
if (Test-Path $ftDir) {
    $binFiles = Get-ChildItem -Path $ftDir -Filter "*.bin" -ErrorAction SilentlyContinue
    foreach ($f in $binFiles) {
        Copy-Item $f.FullName "$projectDir\api\" -Force
        Write-Host "  ✓ $($f.Name)" -ForegroundColor Green
    }
} else {
    Write-Host "  ⚠ Aucun modèle FastText trouvé" -ForegroundColor Yellow
}

# ── 5. Restaurer notebooks ────────────────────────────────────
Write-Host "[ 5/6 ] Restauration notebooks..." -ForegroundColor Yellow
$nbDir = "$BackupDir\notebooks"
if (Test-Path $nbDir) {
    New-Item -ItemType Directory -Force -Path "$projectDir\notebooks" | Out-Null
    $nbFiles = Get-ChildItem -Path $nbDir -Filter "*.ipynb" -ErrorAction SilentlyContinue
    foreach ($f in $nbFiles) {
        Copy-Item $f.FullName "$projectDir\notebooks\" -Force
    }
    Write-Host "  ✓ $($nbFiles.Count) notebooks restaurés" -ForegroundColor Green
} else {
    Write-Host "  ⚠ Dossier notebooks introuvable" -ForegroundColor Yellow
}

# ── 6. Redémarrer l'API pour prendre en compte la DB restaurée
Write-Host "[ 6/6 ] Redémarrage API..." -ForegroundColor Yellow
docker restart onepilot_api
Start-Sleep -Seconds 10
Write-Host "  ✓ API redémarrée" -ForegroundColor Green

# ── Résumé ────────────────────────────────────────────────────
Write-Host ""
Write-Host "══════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "   Restauration terminée !                " -ForegroundColor Cyan
Write-Host "══════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""
Write-Host "   Chat  → http://localhost:3000/chat.html" -ForegroundColor Green
Write-Host "   Index → http://localhost:3000/index.html" -ForegroundColor Green
Write-Host "   API   → http://localhost:8000/docs" -ForegroundColor Green
Write-Host ""
Write-Host "   ⚠ Lance post-start.ps1 pour les modèles voice :" -ForegroundColor Yellow
Write-Host "   .\scripts\post-start.ps1" -ForegroundColor White
Write-Host ""
