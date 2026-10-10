#!/usr/bin/env bash
# Render an explainer HTML to PNG at its full height: render tall, then crop the trailing background.
set -euo pipefail
html=$1; png=${html%.html}.png
google-chrome --headless=new --disable-gpu --hide-scrollbars --window-size=1200,4000 --screenshot="$png" "file://$(realpath "$html")" 2>/dev/null
uv run --with pillow python - "$png" <<'PY'
import sys
from PIL import Image, ImageChops
im = Image.open(sys.argv[1]).convert("RGB")
bg = Image.new("RGB", im.size, im.getpixel((im.width - 1, im.height - 1)))
box = ImageChops.difference(im, bg).getbbox()
im.crop((0, 0, im.width, box[3] + 40)).save(sys.argv[1])
print(sys.argv[1], im.width, box[3] + 40)
PY
