#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat >&2 <<'EOF'
Usage:
  scripts/setup_sklearn_ref.sh --ref REF [options]

Clone (or reuse) a scikit-learn checkout at ./sklearn-src, check out the
requested ref, and install it editable into a Pixi environment. If --remote
points somewhere other than the checkout's current origin, the checkout is
recreated from scratch instead of reusing it.

Options:
  --ref REF             Commit, branch, tag, or remote ref to benchmark. This
                        is fetched from --remote and resolved via FETCH_HEAD.
  --remote URL          scikit-learn git remote URL or local repository path.
                        Default: https://github.com/scikit-learn/scikit-learn.git
  --env NAME            Pixi environment to use. Default: sklearn-dev.
  --recreate            Recreate the checkout even if the remote is unchanged.
  -h, --help            Show this help.

Examples:
  scripts/setup_sklearn_ref.sh --ref main

  scripts/setup_sklearn_ref.sh \
      --remote https://github.com/some-user/scikit-learn.git \
      --ref my-perf-branch
EOF
}

die() {
    echo "error: $*" >&2
    exit 2
}

ref=""
remote="https://github.com/scikit-learn/scikit-learn.git"
pixi_env="sklearn-dev"
checkout="sklearn-src"
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
        --env)
            pixi_env="${2:-}"
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

if [ -e "$checkout" ]; then
    git -C "$checkout" rev-parse --is-inside-work-tree >/dev/null 2>&1 \
        || die "managed path exists but is not a git checkout: $checkout"

    current_remote="$(git -C "$checkout" remote get-url origin 2>/dev/null || true)"
    if [ "$recreate" -eq 1 ] || [ "$current_remote" != "$remote" ]; then
        rm -rf "$checkout"
    elif [ -n "$(git -C "$checkout" status --porcelain --untracked-files=no)" ]; then
        die "managed checkout is dirty: $checkout; commit changes or pass --recreate"
    fi
fi

if [ ! -e "$checkout" ]; then
    git clone --origin origin "$remote" "$checkout"
fi

git -C "$checkout" fetch --tags origin "$ref"
commit="$(git -C "$checkout" rev-parse --verify "FETCH_HEAD^{commit}")"
git -C "$checkout" checkout --detach "$commit"
# The checkout is always left detached (see usage note above), so the requested
# ref name isn't recoverable from git state afterwards. Record it in a sidecar
# file that env.py reads, so dashboards can label builds by ref instead of
# commit. Untracked, so it doesn't affect the checkout's dirty check.
echo "$ref" > "$checkout/.bench-ref"

pixi run -e "$pixi_env" python -m pip install --no-build-isolation --editable "$checkout"
pixi run -e "$pixi_env" python -c \
    'import sklearn; print(f"imported sklearn {sklearn.__version__} from {sklearn.__file__}")'

cat <<EOF

scikit-learn setup complete
  remote:   $remote
  ref:      $ref
  commit:   $commit
  checkout: $checkout
  pixi env: $pixi_env

Run benchmarks with:

  pixi run -e '$pixi_env' python -m sklbench --config configs/all_models_test.py
EOF
