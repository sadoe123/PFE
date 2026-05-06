# ═══════════════════════════════════════════════════════════════
# OnePilot v0 — Post-Start Script
# Lance après chaque : docker compose up -d
# Réinstalle les packages voice qui ne survivent pas au restart
# Usage : .\scripts\post-start.ps1
# ═══════════════════════════════════════════════════════════════

$container = "onepilot_api"
$whisperDir = "/app/models/whisper"
$piperDir   = "/app/models/piper"

Write-Host ""
Write-Host "══════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  OnePilot — Post-Start Voice Setup   " -ForegroundColor Cyan
Write-Host "══════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

# ── 1. Vérifier que le container tourne ──────────────────────
Write-Host "[ 1/5 ] Vérification container..." -ForegroundColor Yellow
$status = docker inspect --format="{{.State.Running}}" $container 2>$null
if ($status -ne "true") {
    Write-Host "  ✗ Container '$container' non démarré — lance d'abord : docker compose up -d" -ForegroundColor Red
    exit 1
}
Write-Host "  ✓ Container actif" -ForegroundColor Green

# ── 2. ffmpeg (système) ───────────────────────────────────────
Write-Host "[ 2/5 ] Installation ffmpeg..." -ForegroundColor Yellow
docker exec -u root $container bash -c "apt-get update -qq && apt-get install -y -qq ffmpeg" 2>$null
$ffmpeg = docker exec $container bash -c "which ffmpeg" 2>$null
if ($ffmpeg) {
    Write-Host "  ✓ ffmpeg OK : $ffmpeg" -ForegroundColor Green
} else {
    Write-Host "  ✗ ffmpeg échoué" -ForegroundColor Red
}

# ── 3. Packages Python voice ──────────────────────────────────
Write-Host "[ 3/5 ] Installation packages Python voice..." -ForegroundColor Yellow
docker exec -u root $container pip install -q setuptools webrtcvad piper-tts vosk soundfile openai-whisper
$vad     = docker exec $container python -c "import webrtcvad; print('ok')" 2>$null
$piper   = docker exec $container python -c "from piper import PiperVoice; print('ok')" 2>$null
$vosk    = docker exec $container python -c "import vosk; print('ok')" 2>$null
$whisper = docker exec $container python -c "import whisper; print('ok')" 2>$null
Write-Host "  webrtcvad : $(if ($vad -eq 'ok') {'✓'} else {'✗'})" -ForegroundColor $(if ($vad -eq 'ok') {'Green'} else {'Red'})
Write-Host "  piper-tts : $(if ($piper -eq 'ok') {'✓'} else {'✗'})" -ForegroundColor $(if ($piper -eq 'ok') {'Green'} else {'Red'})
Write-Host "  vosk      : $(if ($vosk -eq 'ok') {'✓'} else {'✗'})" -ForegroundColor $(if ($vosk -eq 'ok') {'Green'} else {'Red'})
Write-Host "  whisper   : $(if ($whisper -eq 'ok') {'✓'} else {'✗'})" -ForegroundColor $(if ($whisper -eq 'ok') {'Green'} else {'Red'})

# ── 4. Charger le modèle Whisper small ───────────────────────
Write-Host "[ 4/5 ] Chargement modèle Whisper small..." -ForegroundColor Yellow
$whisperModel = docker exec $container python -c "import os; print(os.path.exists('$whisperDir/small.pt'))" 2>$null
if ($whisperModel -eq "True") {
    docker exec $container python -c "import whisper; whisper.load_model('small', download_root='$whisperDir'); print('ok')" 2>$null | Out-Null
    Write-Host "  ✓ Whisper small chargé depuis cache" -ForegroundColor Green
} else {
    Write-Host "  ⚠ Modèle small.pt absent — téléchargement..." -ForegroundColor Yellow
    docker exec $container bash -c "wget -q -c -O $whisperDir/small.pt https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt"
    Write-Host "  ✓ Téléchargement terminé" -ForegroundColor Green
}

# ── 5. Vérifier les voix Piper ────────────────────────────────
Write-Host "[ 5/5 ] Vérification voix Piper..." -ForegroundColor Yellow
$voices = docker exec $container python -c "import os; v=os.listdir('$piperDir') if os.path.exists('$piperDir') else []; print(','.join(v))" 2>$null
if ($voices) {
    Write-Host "  ✓ Voix disponibles : $voices" -ForegroundColor Green
} else {
    Write-Host "  ✗ Aucune voix Piper — téléchargement nécessaire" -ForegroundColor Red
    Write-Host "    Lance : docker exec $container bash -c 'curl -L -o $piperDir/fr_FR-upmc-medium.onnx https://huggingface.co/rhasspy/piper-voices/resolve/main/fr/fr_FR/upmc/medium/fr_FR-upmc-medium.onnx'" -ForegroundColor Gray
}

# ── Résumé final ──────────────────────────────────────────────
Write-Host ""
Write-Host "══════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  Status final                        " -ForegroundColor Cyan
Write-Host "══════════════════════════════════════" -ForegroundColor Cyan
$sttStatus = docker exec $container python -c "
import urllib.request, json
r = urllib.request.urlopen('http://localhost:8000/stt/status')
d = json.loads(r.read())
print('STT:', d.get('available'), '| model:', d.get('model'), '| ffmpeg:', d.get('ffmpeg'), '| vad:', d.get('vad'))
" 2>$null
$ttsStatus = docker exec $container python -c "
import urllib.request, json
r = urllib.request.urlopen('http://localhost:8000/tts/voices')
d = json.loads(r.read())
print('TTS:', d.get('available'), '| voix:', len(d.get('voices', [])))
" 2>$null
Write-Host "  $sttStatus" -ForegroundColor White
Write-Host "  $ttsStatus" -ForegroundColor White
Write-Host ""
Write-Host "  UI  → http://localhost:3000/chat.html" -ForegroundColor Cyan
Write-Host "  API → http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host ""
