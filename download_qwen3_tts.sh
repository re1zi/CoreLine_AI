#!/usr/bin/env bash
# Скачивание Qwen3-TTS-12Hz-1.7B-CustomVoice и токенайзера (9 встроенных голосов).
# Требуется: pip install -U "huggingface_hub[cli]"
# Либо для Китая: pip install -U modelscope и используйте команды из комментариев ниже.

set -e
DIR="${1:-./qwen3_tts_models}"  # папка для моделей (по умолчанию ./qwen3_tts_models)
echo "Скачиваю модели в: $DIR"
mkdir -p "$DIR"
cd "$DIR"

# 1) Токенайзер (общий для всех моделей 12Hz)
echo "=== Токенайзер Qwen3-TTS-Tokenizer-12Hz ==="
hf download Qwen/Qwen3-TTS-Tokenizer-12Hz --local-dir ./Qwen3-TTS-Tokenizer-12Hz

# 2) Модель CustomVoice (9 встроенных голосов: Vivian, Serena, Ryan, Aiden, ...)
echo "=== Модель Qwen3-TTS-12Hz-1.7B-CustomVoice ==="
hf download Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --local-dir ./Qwen3-TTS-12Hz-1.7B-CustomVoice

echo "Готово. Для использования локальных путей задайте переменные перед запуском:"
echo "  export QWEN3_TTS_MODEL=$DIR/Qwen3-TTS-12Hz-1.7B-CustomVoice"
echo "  export TTS_ENGINE=qwen3"
echo ""
echo "Либо оставьте QWEN3_TTS_MODEL=Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice — модель подтянется при первом запуске."
echo "Голос: QWEN3_TTS_SPEAKER=Serena (или Vivian, Ryan, Aiden, Ono_Anna, Sohee, ...)"

# Альтернатива через ModelScope (для пользователей в Китае):
# modelscope download --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --local_dir ./Qwen3-TTS-12Hz-1.7B-CustomVoice
