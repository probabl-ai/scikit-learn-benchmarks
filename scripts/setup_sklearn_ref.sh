#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat >&2 <<'EOF'
Usage:
  scripts/setup_sklearn_ref.sh --ref REF [options]

Fetch a scikit-learn git ref into this repo's managed cache, create a detached
worktree, and install that checkout editable into a Pixi environment.

Options:
  --ref REF             Commit, branch, tag, or remote ref to benchmark. This
                        is fetched from --remote and resolved via FETCH_HEAD.
  --remote URL          scikit-learn git remote URL or local repository path.
                        Default: https://github.com/scikit-learn/scikit-learn.git
  --label LABEL         Stable local name for the managed worktree.
                        Defaults to a sanitized REF.
  --env NAME            Pixi environment to use. Default: sklearn-dev.
  --repo-cache PATH     Managed bare git cache. Default: .bench/scikit-learn.git
  --recreate            Recreate the managed worktree if it already exists.
  -h, --help            Show this help.

Examples:
  scripts/setup_sklearn_ref.sh --ref main --label main

  scripts/setup_sklearn_ref.sh \
      --remote https://github.com/some-user/scikit-learn.git \
      --ref my-perf-branch \
      --label pr
EOF
}

die() {
    echo "error: $*" >&2
    exit 2
}

ref=""
remote="https://github.com/scikit-learn/scikit-learn.git"
label=""
pixi_env="sklearn-dev"
repo_cache=".bench/scikit-learn.git"
recreate=0

while [ "$#" -gt 0 ]; do
    case "$1" in
        --ref)
            ref="${2:-}"
            shift 2
            ;;
        --remote)
            remote="${2:-}"
            shift 2
            ;;
        --label)
            label="${2:-}"
            shift 2
            ;;
        --env)
            pixi_env="${2:-}"
            shift 2
            ;;
        --repo-cache)
            repo_cache="${2:-}"
            shift 2
            ;;
        --recreate)
            recreate=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown argument: $1"
            ;;
    esac
done

[ -n "$ref" ] || die "missing --ref"
[ -n "$remote" ] || die "missing --remote"
[ -n "$repo_cache" ] || die "missing --repo-cache"

if [ ! -d "$repo_cache/objects" ]; then
    mkdir -p "$(dirname "$repo_cache")"
    git init --bare "$repo_cache"
fi

git -C "$repo_cache" fetch --tags "$remote" "$ref"
commit="$(git -C "$repo_cache" rev-parse --verify "FETCH_HEAD^{commit}")"
if [ -z "$label" ]; then
    label="$(printf '%s' "$ref" | tr -c 'A-Za-z0-9._-' '_')"
fi

worktree_root=".bench/sklearn-worktrees"
worktree="$worktree_root/$label"
mkdir -p "$worktree_root"

if [ -e "$worktree" ]; then
    if [ "$recreate" -eq 1 ]; then
        if git -C "$worktree" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
            git -C "$repo_cache" worktree remove --force "$worktree"
        else
            die "refusing to recreate non-worktree path: $worktree"
        fi
    elif ! git -C "$worktree" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        die "managed path exists but is not a git worktree: $worktree"
    elif [ -n "$(git -C "$worktree" status --porcelain --untracked-files=no)" ]; then
        die "managed worktree is dirty: $worktree; commit changes or pass --recreate"
    else
        git -C "$worktree" checkout --detach "$commit"
    fi
fi

if [ ! -e "$worktree" ]; then
    git -C "$repo_cache" worktree add --detach "$worktree" "$commit"
fi

pixi run -e "$pixi_env" python -m pip install --no-build-isolation --editable "$worktree"
pixi run -e "$pixi_env" python -c \
    'import sklearn; print(f"imported sklearn {sklearn.__version__} from {sklearn.__file__}")'

cat <<EOF

scikit-learn setup complete
  remote:   $remote
  ref:      $ref
  commit:   $commit
  worktree: $worktree
  pixi env: $pixi_env

Run benchmarks with:

  pixi run -e '$pixi_env' python -m sklbench --config configs/sklearn.py
EOF
