#!/bin/bash
# Download and prepare all data for this environment (run on a node with internet).
#
# Source: the official release of "Is Complex Query Answering Really Complex?"
# (Gregucci et al., ICML 2025), which bundles the standard BetaE FB15k-237
# query benchmark, the harder FB15k-237+H benchmark and the FB15k-237 triples:
#   https://github.com/april-tools/is-cqa-complex/releases/tag/benchs-1.0
#
# Estimated download: 1.5 GB zip (deleted after preparation).
# Disk after preparation: ~100 MB public data in data/downloads/fb15k237/ and
# ~26 MB hidden test truth in eval/test/data/ (only when eval/test/ is present).
#
# Idempotent: skips everything that is already prepared. The archive contains
# the test answers, so it is removed as soon as both preparation steps finish;
# test answers are only ever written under eval/test/data/.

set -euo pipefail

cd "$(dirname "$0")/.."

URL="https://github.com/april-tools/is-cqa-complex/releases/download/benchs-1.0/iscqa-compl-benchmarks-v1.1.zip"
SHA256="072d6547e0f8bd9362021afb0895f59d097077b1d5043391da88e7eadd027ad0"
PUBLIC=data/downloads/fb15k237
HIDDEN=eval/test/data
ZIPDIR=tmp/downloads
ZIP="$ZIPDIR/iscqa-compl-benchmarks-v1.1.zip"

need_public=0
need_hidden=0
if [ ! -f "$PUBLIC/.complete" ]; then need_public=1; fi
if [ -f eval/test/prepare_hidden.py ] && [ ! -f "$HIDDEN/.complete" ]; then need_hidden=1; fi
if [ "$need_public" -eq 0 ] && [ "$need_hidden" -eq 0 ]; then
  echo "FB15k-237 query data already prepared; nothing to do."
  exit 0
fi

mkdir -p "$ZIPDIR"
if [ ! -f "$ZIP" ]; then
  curl -fL --retry 3 -o "$ZIP.part" "$URL"
  mv "$ZIP.part" "$ZIP"
fi
echo "$SHA256  $ZIP" | sha256sum -c -

if [ "$need_public" -eq 1 ]; then
  python3 -I data/prepare_fb15k237.py --zip "$ZIP" --out "$PUBLIC"
fi
if [ "$need_hidden" -eq 1 ]; then
  python3 -I eval/test/prepare_hidden.py --zip "$ZIP" --out "$HIDDEN" --public "$PUBLIC"
fi
rm -f "$ZIP"
echo "Data prepared."
