#!/usr/bin/env bash
# Installs mmco into ./.venv and puts the `mmco` command on your PATH (~/.local/bin).
set -euo pipefail
cd "$(dirname "$0")"

PYTHON=""
for candidate in python3.14 python3.13 python3.12 python3.11 /opt/homebrew/bin/python3.13 python3; do
  if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import sys; sys.exit(sys.version_info < (3, 11))'; then
    PYTHON="$(command -v "$candidate")"
    break
  fi
done
if [ -z "$PYTHON" ]; then
  echo "Python 3.11+ is required (brew install python@3.13)" >&2
  exit 1
fi

"$PYTHON" -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -e '.[dev]'

mkdir -p "$HOME/.local/bin"
ln -sf "$(pwd)/.venv/bin/mmco" "$HOME/.local/bin/mmco"
case ":$PATH:" in
  *":$HOME/.local/bin:"*) ;;
  *) echo "Add this to ~/.zshrc:  export PATH=\"\$HOME/.local/bin:\$PATH\"" ;;
esac

echo "Installed. Running first-time setup..."
"$HOME/.local/bin/mmco" setup
