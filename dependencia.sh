#!/usr/bin/env bash
set -euo pipefail

# Instala as dependências de sistema (macOS / Homebrew) necessárias para rodar o test.py.
# Pacotes Python ficam por conta do run.sh (venv + pip install).

if ! command -v brew >/dev/null 2>&1; then
  echo "Homebrew não encontrado. Instalando..."
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

echo "Atualizando Homebrew..."
brew update

# python3.12   → interpretador usado pelo run.sh
# portaudio    → dependência nativa do sounddevice (captura de áudio)
# ffmpeg       → usado pelo whisper para decodificar/reamostrar áudio
# blackhole-2ch → driver de áudio virtual (loopback) referenciado por DEVICE no test.py
FORMULAE=(python@3.12 portaudio ffmpeg)
CASKS=(blackhole-2ch)

echo "Instalando formulae: ${FORMULAE[*]}"
brew install "${FORMULAE[@]}"

echo "Instalando casks: ${CASKS[*]}"
brew install --cask "${CASKS[@]}"

echo ""
echo "Dependências de sistema instaladas."
echo "Próximos passos:"
echo "  1) Abra 'Configuração de Áudio e MIDI' e crie um 'Dispositivo de Múltiplas Saídas'"
echo "     combinando seus alto-falantes + BlackHole 2ch, e selecione-o como saída do sistema."
echo "  2) Rode: ./run.sh"
