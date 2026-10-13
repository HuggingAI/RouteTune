#!/bin/sh
set -eu

if [ "${1:-}" = "routetune" ] && [ "${2:-}" = "serve" ]; then
  bundle_dir="/app/artifacts/bootstrap/label-bundle"
  if [ ! -f "$bundle_dir/label-bundle.joblib" ]; then
    routetune bootstrap-labels --output "$bundle_dir"
  fi
fi

exec "$@"
