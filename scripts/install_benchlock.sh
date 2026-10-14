#!/usr/bin/env bash
# Installs scripts/benchlock as ~/.local/bin/benchlock on this machine and on
# the remote benchmark machines. Rerun it after changing scripts/benchlock.
#
# Usage: scripts/install_benchlock.sh [HOST ...]   (default: local intel-gnr intel-laptop)
set -euo pipefail

src="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/benchlock"
hosts=("$@")
[ ${#hosts[@]} -gt 0 ] || hosts=(local intel-gnr intel-laptop)

for h in "${hosts[@]}"; do
    if [ "$h" = local ]; then
        install -D -m 755 "$src" ~/.local/bin/benchlock
        command -v benchlock >/dev/null || echo "local: ~/.local/bin is not on your PATH"
    else
        ssh "$h" 'mkdir -p ~/.local/bin && cat > ~/.local/bin/benchlock && chmod 755 ~/.local/bin/benchlock' <"$src"
        # `ssh host cmd` runs a non-interactive shell, which doesn't always
        # put ~/.local/bin on the PATH (it doesn't on intel-laptop).
        if ! ssh "$h" 'command -v benchlock' >/dev/null; then
            echo "$h: benchlock is not on the PATH of non-interactive ssh commands, run once:"
            echo "  ssh -t $h 'sudo ln -sf ~/.local/bin/benchlock /usr/local/bin/benchlock'"
        fi
    fi
    echo "$h: installed"
done
