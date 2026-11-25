#!/usr/bin/env bash
# Build the shareable archive of this project.
#
# The exclusions are not tidiness. data/sources and data/raw hold corpora that
# their licences let you keep and not pass on, so an archive carrying them is a
# licence breach in a file somebody will email.
set -euo pipefail

cd "$(dirname "$0")/.."
name="FinTextIntelligence"
out="$name/$name.zip"
rm -f "$out"

zip -qr "$out" "$name" \
  -x "$name/.git/*" \
     "$name/.mypy_cache/*" "$name/.pytest_cache/*" "$name/.ruff_cache/*" \
     "*__pycache__*" "*node_modules/*" \
     "$name/data/sources/*" "$name/data/raw/*" \
     "$name/data/interim/*" "$name/data/processed/*" \
     "$name/artifacts/models/*" "$name/reports/*" \
     "$name/.cl*/*" "*.zip" "*.env" "*.token" \
     "*pytorch_model.bin" "*.safetensors" "*.pt" "*.ckpt" "*.onnx" "*tokenizer.json"

echo "wrote $out ($(du -h "$out" | cut -f1))"

# Prove the things that must not travel did not travel. Weights are excluded by
# path AND by name: a 1 GB model unzipped into notebooks/ slipped past a
# path-only rule once, and the only symptom was an archive that had quietly
# grown from 356K to 832M.
if unzip -l "$out" | grep -qE "data/(sources|raw)/"; then
  echo "ERROR: licensed corpora are in the archive" >&2
  exit 1
fi
if unzip -l "$out" | grep -qE "pytorch_model\.bin|\.safetensors|\.onnx$"; then
  echo "ERROR: model weights are in the archive" >&2
  exit 1
fi

size_kb=$(du -k "$out" | cut -f1)
if [ "$size_kb" -gt 20480 ]; then
  echo "ERROR: archive is ${size_kb}KB; something large got in" >&2
  exit 1
fi
echo "checked: no licensed data, no weights, size sane"
