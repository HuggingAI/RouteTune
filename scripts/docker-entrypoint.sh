#!/bin/sh
set -eu

if [ "${1:-}" = "routetune" ] && [ "${2:-}" = "serve" ]; then
  bundle_dir="/app/artifacts/bootstrap/label-bundle"
  if [ ! -f "$bundle_dir/label-bundle.joblib" ]; then
    python -m routetune_core.cli bootstrap-labels --output "$bundle_dir"
  fi
fi

if [ "${1:-}" = "routetune" ]; then
  shift
  set -- python -m routetune_core.cli "$@"
fi

exec "$@"
