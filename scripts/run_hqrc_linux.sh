#!/bin/sh
set -eu

action=${1:?usage: run_hqrc_linux.sh import|proposal|smoke|paper|resume [arguments...]}
shift
accelerator=${HQRC_ACCELERATOR:-cuda}

for argument in "$@"; do
    case "$argument" in
        --profile|--profile=*) echo "profile is owned by the launcher action" >&2; exit 2 ;;
        --accelerator|--accelerator=*) echo "accelerator is owned by the launcher control" >&2; exit 2 ;;
    esac
done

uv sync --project hqrc_v3 --extra accelerator --locked

case "$action" in
    import) exec uv run --project hqrc_v3 --extra accelerator --locked hqrc import-paper-source "$@" ;;
    proposal) exec uv run --project hqrc_v3 --extra accelerator --locked hqrc run-loeo-accelerated --profile paper --accelerator "$accelerator" "$@" ;;
    smoke) exec uv run --project hqrc_v3 --extra accelerator --locked hqrc run-loeo-accelerated --profile smoke --accelerator "$accelerator" --draws 4 --tune 4 --chains 4 --target-accept 0.9 "$@" ;;
    paper|resume) exec uv run --project hqrc_v3 --extra accelerator --locked hqrc run-loeo-accelerated --profile paper --accelerator "$accelerator" "$@" ;;
    *) echo "unknown action: $action" >&2; exit 2 ;;
esac
