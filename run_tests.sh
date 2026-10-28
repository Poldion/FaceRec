#!/usr/bin/env bash
# Tests der Gesichtserkennung im App-Image (gleiche Python-/Paketversionen wie im Betrieb).
# Braucht nur Docker; der Erkenner wird nachgebaut, echte Daten bleiben unberuehrt.
#   facerec/run_tests.sh [pytest-Optionen, z.B. -k assign -x]
set -euo pipefail
cd "$(dirname "$0")"
docker build -q -t facerec:test . >/dev/null
docker run --rm -v "$PWD":/src:ro -e PYTHONDONTWRITEBYTECODE=1 -w /tmp facerec:test \
  sh -c 'pip install -q --disable-pip-version-check --no-cache-dir pytest==8.3.4 >/dev/null 2>&1 &&
         cp -r /src /tmp/facerec && cd /tmp/facerec && python -m pytest -q -p no:cacheprovider tests "$@"' sh "$@"
