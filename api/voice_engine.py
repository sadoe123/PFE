"""
OnePilot – Voice Engine §2.3.2
Speech-to-Text via Whisper (on-premise)
- Batch recognition
- VAD (Voice Activity Detection)
- Normalisation numérique
- Custom vocabulary métier
"""
from __future__ import annotations

import io
import logging
import os
import re
import tempfile
import time
from typing import Optional

logger = logging.getLogger(__name__)

# ── VAD Configuration ────────────────────────────────────────────────────────
VAD_AGGRESSIVENESS = int(os.environ.get("VAD_AGGRESSIVENESS", "2"))  # 0-3
VAD_SAMPLE_RATE    = 16000  # Hz — requis par webrtcvad
VAD_FRAME_MS       = 30     # ms par frame (10, 20 ou 30)

# ── Configuration ───────────────────────────────────────────────────────────
WHISPER_MODEL  = os.environ.get("WHISPER_MODEL", "small")   # tiny/base/small/medium
WHISPER_LANG   = os.environ.get("WHISPER_LANG",  "fr")     # fr/en/auto

# Vocabulaire métier custom — améliore la précision sur termes spécifiques
BUSINESS_VOCABULARY = [
    # Tables SXA
    "SI_Trésorerie", "GS_ACC", "SI_Bancaire", "FINANCEMENT_BI",
    "AA_AALTUSRTL", "GS_GLACC", "RC_BAL",
    # Tables Northwind
    "Customers", "Orders", "Products", "Employees", "Categories",
    "Suppliers", "Shippers", "Territories",
    # Termes métier
    "AMOUNTI", "TRNDATE", "ISDEBITI", "Banque", "Société",
    "CustomerID", "EmployeeID", "ProductID", "OrderID",
    "Freight", "UnitPrice", "UnitsInStock",
    # Commandes SQL
    "total", "cumulatif", "GROUP BY", "HAVING", "jointure",
    "top cinq", "top dix", "top vingt",
    # Commandes vocales OnePilot
    "répète", "reformule", "approfondi", "plus de détails",
    "montre le dashboard", "affiche le graphe", "exporte",
    "nouvelle conversation", "annule", "historique",
    "répéter", "reformuler", "approfondir",
]

# ── Normalisation voix → texte ───────────────────────────────────────────────
NUMBER_WORDS = {
    "zéro": "0", "zero": "0",
    "un": "1", "une": "1",
    "deux": "2", "trois": "3", "quatre": "4", "cinq": "5",
    "six": "6", "sept": "7", "huit": "8", "neuf": "9",
    "dix": "10", "onze": "11", "douze": "12", "treize": "13",
    "quatorze": "14", "quinze": "15", "seize": "16",
    "vingt": "20", "trente": "30", "quarante": "40",
    "cinquante": "50", "soixante": "60", "cent": "100",
    "mille": "1000", "million": "1000000",
    "deux mille": "2000", "deux mille vingt": "2020",
    "deux mille vingt et un": "2021", "deux mille vingt deux": "2022",
    "deux mille vingt trois": "2023", "deux mille vingt quatre": "2024",
    "deux mille vingt cinq": "2025",
}

# Commandes vocales spéciales → actions
VOICE_COMMANDS = {
    # Répétition
    "répète":               "repeat_last",
    "repete":               "repeat_last",
    "redis":                "repeat_last",
    # Approfondissement
    "plus de détails":      "more_details",
    "plus de details":      "more_details",
    "approfondi":           "more_details",
    "explique plus":        "more_details",
    # Reformulation
    "reformule":            "rephrase",
    "explique autrement":   "rephrase",
    "explique différemment":"rephrase",
    # Dashboard
    "montre le dashboard":  "show_dashboard",
    "affiche le dashboard": "show_dashboard",
    "montre le graphe":     "show_chart",
    "affiche le graphe":    "show_chart",
    "visualise":            "show_chart",
    # Export
    "envoie par email":     "export_email",
    "exporte":              "export_data",
    "télécharge":           "export_data",
    "telecharge":           "export_data",
    # Navigation
    "historique":           "show_history",
    "affiche l historique": "show_history",
    "affiche historique":   "show_history",
    "question suivante":    "next_question",
    "suivante":             "next_question",
    "prochaine":            "next_question",
    "question precedente":  "prev_question",
    "question précédente":  "prev_question",
    "precedente":           "prev_question",
    "retour":               "prev_question",
    # Conversation
    "nouvelle conversation":"new_chat",
    "nouvelle conv":        "new_chat",
    "effacer":              "clear_chat",
    "annule":               "cancel",
    "stop":                 "cancel",
}


def normalize_voice_text(text: str) -> str:
    """
    Normalise le texte transcrit par Whisper :
    - Nombres écrits → chiffres
    - Corrections orthographiques courantes
    - Suppressions de bruits parasites
    """
    if not text:
        return text

    result = text.strip()

    # Nombres composés (ordre décroissant de longueur)
    for word, num in sorted(NUMBER_WORDS.items(), key=lambda x: -len(x[0])):
        result = re.sub(r'\b' + re.escape(word) + r'\b', num, result, flags=re.IGNORECASE)

    # Corrections communes STT
    corrections = {
        "si trésorerie":   "SI_Trésorerie",
        "si tresorerie":   "SI_Trésorerie",
        "gs acc":          "GS_ACC",
        "group by":        "GROUP BY",
        "order by":        "ORDER BY",
        "sum":             "SUM",
        "count":           "COUNT",
    }
    for wrong, correct in corrections.items():
        result = re.sub(r'\b' + re.escape(wrong) + r'\b', correct, result, flags=re.IGNORECASE)

    # Supprime les hésitations vocales
    result = re.sub(r'\b(euh|hum|hmm|ah|oh)\b', '', result, flags=re.IGNORECASE)
    result = re.sub(r'\s+', ' ', result).strip()

    return result


def detect_voice_command(text: str) -> Optional[str]:
    """
    Détecte si le texte est une commande vocale spéciale.
    Retourne l'action ou None si c'est une question normale.
    """
    import unicodedata
    def _norm(s):
        # Normalise accents + minuscules
        s = s.lower().strip()
        s = ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')
        # Supprime ponctuation
        s = ''.join(c if c.isalnum() or c == ' ' else ' ' for c in s)
        return ' '.join(s.split())

    text_norm = _norm(text)

    for trigger, action in VOICE_COMMANDS.items():
        trigger_norm = _norm(trigger)
        if trigger_norm in text_norm:
            logger.info(f"[STT] Commande détectée: '{trigger}' → {action}")
            return action

    # Correspondance partielle pour les commandes courtes
    words = set(text_norm.split())
    cmd_map = {
        'repete': 'repeat_last', 'repeter': 'repeat_last', 'redis': 'repeat_last', 'repetit': 'repeat_last', 'repeti': 'repeat_last', 'repet': 'repeat_last',
        'reformule': 'rephrase', 'reformuler': 'rephrase',
        'approfondi': 'more_details', 'detaille': 'more_details',
        'dashboard': 'show_dashboard', 'graphe': 'show_chart',
        'exporte': 'export_data', 'telecharge': 'export_data',
        'historique': 'show_history',
        'annule': 'cancel', 'stop': 'cancel',
    }
    for word in words:
        if word in cmd_map:
            action = cmd_map[word]
            logger.info(f"[STT] Commande partielle: '{word}' → {action}")
            return action

    return None


class VoiceActivityDetector:
    """
    VAD via webrtcvad — détecte automatiquement début/fin de parole.
    Aggressiveness : 0 (permissif) → 3 (strict)
    """

    def __init__(self, aggressiveness: int = VAD_AGGRESSIVENESS):
        self.aggressiveness = aggressiveness
        self._vad = None

    def _get_vad(self):
        if self._vad is None:
            import webrtcvad
            self._vad = webrtcvad.Vad(self.aggressiveness)
        return self._vad

    def is_speech(self, audio_chunk: bytes, sample_rate: int = VAD_SAMPLE_RATE) -> bool:
        """Détecte si un chunk audio contient de la parole."""
        try:
            vad = self._get_vad()
            return vad.is_speech(audio_chunk, sample_rate)
        except Exception:
            return True  # Par défaut considère tout comme parole

    def filter_silence(
        self,
        audio_bytes: bytes,
        sample_rate: int = VAD_SAMPLE_RATE,
        frame_ms: int = VAD_FRAME_MS,
    ) -> bytes:
        """
        Filtre les silences d'un fichier audio PCM 16-bit.
        Retourne seulement les frames contenant de la parole.
        """
        try:
            import array
            frame_size = int(sample_rate * frame_ms / 1000) * 2  # 2 bytes per sample
            vad = self._get_vad()
            speech_frames = []

            for i in range(0, len(audio_bytes) - frame_size, frame_size):
                frame = audio_bytes[i:i + frame_size]
                if len(frame) == frame_size:
                    try:
                        if vad.is_speech(frame, sample_rate):
                            speech_frames.append(frame)
                    except Exception:
                        speech_frames.append(frame)

            if speech_frames:
                return b"".join(speech_frames)
            return audio_bytes  # Si rien détecté → retourne tout

        except Exception as e:
            logger.warning(f"[VAD] Erreur filtre: {e}")
            return audio_bytes


# ── Singleton VAD ────────────────────────────────────────────────────────────
_vad_instance = None

def get_vad() -> VoiceActivityDetector:
    global _vad_instance
    if _vad_instance is None:
        _vad_instance = VoiceActivityDetector()
    return _vad_instance


class WhisperSTT:
    """
    Transcription audio via Whisper (on-premise).
    Supporte : batch (fichier complet) et normalisation automatique.
    """

    def __init__(self, model_name: str = WHISPER_MODEL, language: str = WHISPER_LANG):
        self.model_name = model_name
        self.language   = language
        self._model     = None

    def _load_model(self):
        """Charge le modèle Whisper (lazy loading)."""
        if self._model is None:
            try:
                import whisper
                import os
                # Force le cache dans /tmp accessible à tous
                # Chemin persistant — volume Docker /app/models/whisper
                whisper_dir = os.environ.get("WHISPER_CACHE_DIR", "/app/models/whisper")
                os.makedirs(whisper_dir, exist_ok=True)
                os.environ["XDG_CACHE_HOME"] = whisper_dir
                logger.info(f"[STT] Chargement modèle Whisper '{self.model_name}' depuis {whisper_dir}...")
                self._model = whisper.load_model(
                    self.model_name,
                    download_root=whisper_dir
                )
                logger.info(f"[STT] Modèle Whisper '{self.model_name}' chargé ✅")
            except Exception as e:
                logger.error(f"[STT] Erreur chargement Whisper: {e}")
                raise
        return self._model

    def transcribe(self, audio_bytes: bytes, filename: str = "audio.webm") -> dict:
        """
        Transcrit un fichier audio (bytes) en texte.
        Retourne {text, language, duration_ms, normalized, command}
        """
        t0 = time.time()

        try:
            model = self._load_model()

            # Sauvegarde temporaire du fichier audio
            suffix = os.path.splitext(filename)[1] or ".webm"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name

            try:
                # ── Preprocessing audio via ffmpeg ──────────────────────
                import subprocess as _sp
                preprocessed_path = tmp_path + "_clean.wav"
                try:
                    _sp.run([
                        "ffmpeg", "-y", "-i", tmp_path,
                        "-af", "loudnorm,highpass=f=200,lowpass=f=3000",
                        "-ar", "16000",   # 16kHz requis par Whisper
                        "-ac", "1",       # mono
                        preprocessed_path
                    ], capture_output=True, timeout=30, check=True)
                    transcribe_path = preprocessed_path
                    logger.info("[STT] Preprocessing audio OK (loudnorm + filtres)")
                except Exception as e:
                    logger.warning(f"[STT] Preprocessing skip: {e}")
                    transcribe_path = tmp_path

                # Options Whisper
                options = {
                    "language":            self.language if self.language != "auto" else None,
                    "task":                "transcribe",
                    "verbose":             False,
                    "fp16":                False,   # CPU compatible
                    "initial_prompt":      " ".join(BUSINESS_VOCABULARY[:20]),  # vocabulaire métier
                }

                result = model.transcribe(transcribe_path, **{k: v for k, v in options.items() if v is not None})
                raw_text  = result.get("text", "").strip()
                language  = result.get("language", self.language)

            finally:
                os.unlink(tmp_path)
                try:
                    if os.path.exists(tmp_path + "_clean.wav"):
                        os.unlink(tmp_path + "_clean.wav")
                except: pass

            # Normalisation
            normalized = normalize_voice_text(raw_text)

            # Détection commande vocale
            command = detect_voice_command(normalized)

            ms = int((time.time() - t0) * 1000)
            logger.info(f"[STT] Transcription: '{normalized[:60]}' ({ms}ms)")

            return {
                "text":        normalized,
                "raw_text":    raw_text,
                "language":    language,
                "duration_ms": ms,
                "command":     command,
                "model":       self.model_name,
            }

        except Exception as e:
            logger.error(f"[STT] Erreur transcription: {e}")
            return {
                "text":        "",
                "raw_text":    "",
                "language":    self.language,
                "duration_ms": int((time.time() - t0) * 1000),
                "command":     None,
                "error":       str(e),
                "model":       self.model_name,
            }


# ── Piper TTS Configuration ─────────────────────────────────────────────────
PIPER_VOICES_DIR  = os.environ.get("PIPER_VOICES_DIR", "/tmp/piper/voices")
PIPER_DEFAULT_VOICE = os.environ.get("PIPER_DEFAULT_VOICE", "fr_FR-upmc-medium")

AVAILABLE_VOICES = {
    "fr_female":  "fr_FR-upmc-medium",
    "fr_default": "fr_FR-upmc-medium",
}


class PiperTTS:
    """
    Text-to-Speech via Piper (on-premise).
    Supporte : SSML basique, vitesse ajustable, voix multiples.
    """

    def __init__(self, voices_dir: str = PIPER_VOICES_DIR):
        self.voices_dir = voices_dir
        self._voices = {}  # cache des modèles chargés

    def _load_voice(self, voice_name: str):
        """Charge une voix Piper (lazy loading avec cache)."""
        _aliases = {"female":"fr_FR-upmc-medium","male":"fr_FR-gilles-low",
                    "femme":"fr_FR-upmc-medium","homme":"fr_FR-gilles-low"}
        voice_name = _aliases.get(voice_name, voice_name)
        if voice_name not in self._voices:
            onnx_path = os.path.join(self.voices_dir, f"{voice_name}.onnx")
            if not os.path.exists(onnx_path):
                raise FileNotFoundError(f"Voix non trouvée: {onnx_path}")
            try:
                from piper import PiperVoice
                logger.info(f"[TTS] Chargement voix '{voice_name}'...")
                self._voices[voice_name] = PiperVoice.load(onnx_path)
                logger.info(f"[TTS] Voix '{voice_name}' chargée ✅")
            except Exception as e:
                logger.error(f"[TTS] Erreur chargement voix: {e}")
                raise
        return self._voices[voice_name]

    def _strip_ssml(self, text: str) -> str:
        """Supprime les balises SSML pour Piper (qui ne supporte pas SSML natif)."""
        import re
        # Traite les balises SSML basiques
        text = re.sub(r'<break[^>]*/>', ' ', text)
        text = re.sub(r'<emphasis[^>]*>(.*?)</emphasis>', r'', text)
        text = re.sub(r'<speak[^>]*>(.*?)</speak>', r'', text, flags=re.DOTALL)
        text = re.sub(r'<[^>]+>', '', text)
        return text.strip()

    def simplify_for_voice(self, text: str) -> str:
        """Simplifie le texte pour TTS — supprime emojis, Markdown, tables, backticks."""
        import re, unicodedata
        text = re.sub(r'''```[\s\S]*?```''', ' ', text)
        text = re.sub(r'`[^`\n]+`', '', text)
        def _rm_emoji(s):
            out = []
            for ch in s:
                cp = ord(ch)
                cat = unicodedata.category(ch)
                if cat.startswith('L') or cat.startswith('N') or ch in ' \n.,;:!?()\'-':
                    out.append(ch)
                elif 0x1F000<=cp<=0x1FFFF or 0x2600<=cp<=0x27FF:
                    out.append(' ')
            return ''.join(out)
        text = _rm_emoji(text)
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        text = re.sub(r'#{1,6}\s+', '', text)
        text = re.sub(r'\|[^\n]+', '', text)
        text = re.sub(r'[`~^_<>{}\[\]]', '', text)
        text = re.sub(r'\n\s*[A-Z_]+\.[A-Z_]+[^\n]*', '', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def pronunciation_hints(self, text: str) -> str:
        """
        Convertit les abréviations numériques en texte prononçable.
        Exemples: 2.5M → deux virgule cinq millions, 15% → quinze pour cent
        """
        import re

        # Milliards
        def _replace_b(m):
            val = m.group(1).replace(',', '.').replace(' ', '')
            try:
                n = float(val)
                int_part = int(n)
                dec_part = round((n - int_part) * 10)
                if dec_part:
                    return f"{int_part} virgule {dec_part} milliards"
                return f"{int_part} milliards"
            except: return m.group(0)

        # Millions
        def _replace_m(m):
            val = m.group(1).replace(',', '.').replace(' ', '')
            try:
                n = float(val)
                int_part = int(n)
                dec_part = round((n - int_part) * 10)
                if dec_part:
                    return f"{int_part} virgule {dec_part} millions"
                return f"{int_part} millions"
            except: return m.group(0)

        # Milliers
        def _replace_k(m):
            val = m.group(1).replace(',', '.').replace(' ', '')
            try:
                n = float(val)
                int_part = int(n)
                return f"{int_part} mille"
            except: return m.group(0)

        text = re.sub(r'([\d,. ]+)\s*[Bb](?!\w)', _replace_b, text)
        text = re.sub(r'([\d,. ]+)\s*[Mm](?!\w)', _replace_m, text)
        text = re.sub(r'([\d,. ]+)\s*[Kk](?!\w)', _replace_k, text)
        # Pourcentages
        text = re.sub(r'(\d+(?:\.\d+)?)\s*%', lambda m: f"{m.group(1)} pour cent", text)
        # Euros
        text = re.sub(r'(\d+(?:[,. ]\d+)?)\s*€', lambda m: f"{m.group(1)} euros", text)
        # ms → millisecondes
        text = re.sub(r'(\d+)\s*ms', lambda m: f"{m.group(1)} millisecondes", text)
        # SQL → "S Q L"
        text = text.replace('SQL', 'S Q L')
        text = text.replace('NLU', 'N L U')
        text = text.replace('API', 'A P I')
        text = text.replace('KPI', 'K P I')
        return text

    def vocal_summary(self, text: str, max_chars: int = 600) -> str:
        """
        Génère un résumé vocal pour les longues réponses.
        Si le texte > max_chars après simplification, extrait les points clés.
        """
        simplified = self.simplify_for_voice(text)
        simplified = self.pronunciation_hints(simplified)

        # Si le texte original était long mais simplifié en court → ajoute conclusion
        original_len = len(text)
        if len(simplified) <= max_chars:
            if original_len > 500:
                return simplified.rstrip('.:') + ". La réponse complète est affichée à l'écran."
            return simplified

        # Extrait les 2-3 premières phrases significatives
        import re
        sentences = re.split(r'(?<=[.!?])\s+', simplified)
        summary_parts = []
        total = 0
        for sent in sentences:
            sent = sent.strip()
            if not sent or len(sent) < 10:
                continue
            if total + len(sent) > max_chars:
                break
            summary_parts.append(sent)
            total += len(sent)

        if summary_parts:
            summary = ' '.join(summary_parts)
            # Ajoute mention du résumé
            if len(simplified) > max_chars * 1.5:
                summary += ". La réponse complète est affichée à l'écran."
            return summary

        # Fallback : tronque proprement à la dernière phrase
        truncated = simplified[:max_chars]
        last_dot = max(truncated.rfind('.'), truncated.rfind('!'), truncated.rfind('?'))
        if last_dot > max_chars // 2:
            return truncated[:last_dot + 1] + " Suite à l'écran."
        return truncated + "... Suite à l'écran."

    def synthesize(
        self,
        text: str,
        voice: str = None,
        speed: float = 1.0,
    ) -> bytes:
        """
        Synthétise du texte en audio WAV.
        Retourne les bytes WAV.
        """
        import io
        import wave
        import time

        voice_name = voice or PIPER_DEFAULT_VOICE
        t0 = time.time()

        # Pipeline de préparation vocale
        # 1. Simplification Markdown + code blocks
        clean_text = self.simplify_for_voice(text)
        # 2. Nettoyage SSML résiduel
        clean_text = self._strip_ssml(clean_text)
        # 3. Résumé vocal si texte long
        clean_text = self.vocal_summary(clean_text, max_chars=600)
        # 4. Pronunciation hints
        clean_text = self.pronunciation_hints(clean_text)

        if not clean_text:
            return b""

        try:
            piper_voice = self._load_voice(voice_name)

            # Piper retourne un itérable de AudioChunk
            audio_chunks = list(piper_voice.synthesize(clean_text))

            if not audio_chunks:
                raise ValueError("Aucun audio généré")

            # Récupère les infos audio du premier chunk
            first = audio_chunks[0]
            sample_rate  = first.sample_rate
            sample_width = first.sample_width
            channels     = first.sample_channels
            logger.info(f"[TTS] Audio params: rate={sample_rate} width={sample_width} channels={channels}")

            # Assemble tous les chunks en WAV
            buf = io.BytesIO()
            with wave.open(buf, 'wb') as wav_file:
                wav_file.setnchannels(channels)
                wav_file.setsampwidth(sample_width)
                wav_file.setframerate(sample_rate)
                for chunk in audio_chunks:
                    wav_file.writeframes(chunk.audio_int16_bytes)

            wav_bytes = buf.getvalue()
            ms = int((time.time() - t0) * 1000)
            logger.info(f"[TTS] '{clean_text[:40]}' → {len(wav_bytes)} bytes ({ms}ms)")
            return wav_bytes

        except Exception as e:
            logger.error(f"[TTS] Erreur synthèse: {e}")
            raise

    def list_voices(self) -> list:
        """Retourne la liste des voix disponibles."""
        voices = []
        if os.path.exists(self.voices_dir):
            for f in os.listdir(self.voices_dir):
                if f.endswith('.onnx'):
                    voices.append(f.replace('.onnx', ''))
        return voices


# ── Singleton TTS ─────────────────────────────────────────────────────────────
_tts_instance = None

def get_tts_engine() -> PiperTTS:
    global _tts_instance
    if _tts_instance is None:
        _tts_instance = PiperTTS()
    return _tts_instance


def check_piper_available() -> dict:
    """Vérifie si Piper TTS est disponible."""
    try:
        from piper import PiperVoice
        tts = get_tts_engine()
        voices = tts.list_voices()
        return {
            "available": len(voices) > 0,
            "voices":    voices,
            "default":   PIPER_DEFAULT_VOICE,
            "voices_dir": PIPER_VOICES_DIR,
        }
    except ImportError:
        return {"available": False, "error": "piper-tts non installé"}
    except Exception as e:
        return {"available": False, "error": str(e)}


# ── Vosk Streaming STT ──────────────────────────────────────────────────────
VOSK_MODEL_PATH = os.environ.get("VOSK_MODEL_PATH", "/tmp/vosk/vosk-model-small-fr-0.22")
VOSK_SAMPLE_RATE = 16000


class VoskSTT:
    """
    STT streaming via Vosk — transcription mot par mot en temps réel.
    Utilisé pour le preview temps réel pendant l'enregistrement.
    """

    def __init__(self, model_path: str = VOSK_MODEL_PATH):
        self.model_path = model_path
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                from vosk import Model
                import logging as _logging
                _logging.getLogger("vosk").setLevel(_logging.ERROR)
                logger.info(f"[VoskSTT] Chargement modèle: {self.model_path}")
                self._model = Model(self.model_path)
                logger.info("[VoskSTT] Modèle chargé ✅")
            except Exception as e:
                logger.error(f"[VoskSTT] Erreur chargement: {e}")
                raise
        return self._model

    def transcribe_pcm(self, pcm_bytes: bytes, sample_rate: int = VOSK_SAMPLE_RATE) -> str:
        """
        Transcrit des bytes PCM 16-bit mono en texte.
        Retourne le texte final.
        """
        try:
            from vosk import KaldiRecognizer
            model = self._load_model()
            rec = KaldiRecognizer(model, sample_rate)
            rec.SetWords(True)

            # Traitement par chunks de 4000 bytes
            chunk_size = 4000
            results = []
            for i in range(0, len(pcm_bytes), chunk_size):
                chunk = pcm_bytes[i:i + chunk_size]
                if rec.AcceptWaveform(chunk):
                    import json
                    result = json.loads(rec.Result())
                    if result.get("text"):
                        results.append(result["text"])

            # Résultat final
            import json
            final = json.loads(rec.FinalResult())
            if final.get("text"):
                results.append(final["text"])

            return " ".join(results).strip()

        except Exception as e:
            logger.error(f"[VoskSTT] Erreur transcription: {e}")
            return ""

    def transcribe_audio_file(self, audio_bytes: bytes, filename: str = "audio.webm") -> dict:
        """
        Transcrit un fichier audio via conversion PCM puis Vosk.
        """
        import subprocess
        import tempfile
        import time
        t0 = time.time()

        try:
            # Convertit webm/mp3/etc → PCM 16-bit mono via ffmpeg
            suffix = os.path.splitext(filename)[1] or ".webm"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_in:
                tmp_in.write(audio_bytes)
                tmp_in_path = tmp_in.name

            tmp_out_path = tmp_in_path + ".pcm"
            try:
                subprocess.run([
                    "ffmpeg", "-y", "-i", tmp_in_path,
                    "-ar", str(VOSK_SAMPLE_RATE),
                    "-ac", "1",
                    "-f", "s16le",
                    tmp_out_path
                ], capture_output=True, timeout=30)

                with open(tmp_out_path, "rb") as f:
                    pcm_bytes = f.read()

                text = self.transcribe_pcm(pcm_bytes)
                normalized = normalize_voice_text(text)
                command = detect_voice_command(normalized)

                ms = int((time.time() - t0) * 1000)
                return {
                    "text":        normalized,
                    "raw_text":    text,
                    "language":    "fr",
                    "duration_ms": ms,
                    "command":     command,
                    "model":       "vosk-small-fr",
                    "engine":      "vosk",
                }
            finally:
                for p in [tmp_in_path, tmp_out_path]:
                    try: os.unlink(p)
                    except: pass

        except Exception as e:
            logger.error(f"[VoskSTT] Erreur: {e}")
            return {
                "text": "", "raw_text": "", "language": "fr",
                "duration_ms": 0, "command": None,
                "error": str(e), "engine": "vosk",
            }


# ── Singleton Vosk ───────────────────────────────────────────────────────────
_vosk_instance = None

def get_vosk_engine() -> VoskSTT:
    global _vosk_instance
    if _vosk_instance is None:
        _vosk_instance = VoskSTT()
    return _vosk_instance


def check_vosk_available() -> dict:
    """Vérifie si Vosk est disponible."""
    try:
        import vosk
        model_exists = os.path.exists(VOSK_MODEL_PATH)
        return {
            "available":   model_exists,
            "model_path":  VOSK_MODEL_PATH,
            "model_exists": model_exists,
        }
    except ImportError:
        return {"available": False, "error": "vosk non installé"}


# ── Singleton global ─────────────────────────────────────────────────────────
_stt_instance: Optional[WhisperSTT] = None

def get_stt_engine() -> WhisperSTT:
    global _stt_instance
    if _stt_instance is None:
        _stt_instance = WhisperSTT()
    return _stt_instance


def check_whisper_available() -> dict:
    """Vérifie si Whisper est disponible."""
    try:
        import whisper
        models = ["tiny", "base", "small", "medium", "large"]
        vad_ok = False
        try:
            import webrtcvad
            vad_ok = True
        except ImportError:
            pass
        return {
            "available":    True,
            "model":        WHISPER_MODEL,
            "language":     WHISPER_LANG,
            "models":       models,
            "ffmpeg":       _check_ffmpeg(),
            "vad":          vad_ok,
        }
    except ImportError:
        return {"available": False, "error": "openai-whisper non installé"}


def _check_ffmpeg() -> bool:
    """Vérifie si ffmpeg est disponible."""
    import subprocess
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=5)
        return True
    except Exception:
        return False