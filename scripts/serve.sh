#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v bundle >/dev/null 2>&1; then
  echo "Bundler is required. See README.md for setup instructions." >&2
  exit 1
fi

if ! bundle check >/dev/null; then
  echo "Install dependencies with 'bundle install' before starting the preview." >&2
  exit 1
fi

exec bundle exec jekyll serve --host 127.0.0.1 --baseurl "" --livereload --drafts --future --trace "$@"
