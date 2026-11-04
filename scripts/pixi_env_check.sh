#!/usr/bin/env bash
# Shared helper, sourced by run.sh and dispatch.sh: checks whether a name is
# a pixi environment actually defined in this repo's pixi.toml, so a typo'd
# or removed env name fails fast with a clear message instead of failing
# later (after e.g. a slow sklearn ref checkout, or a CI dispatch).

pixi_env_exists() {
    local name="$1"
    if [ -z "${_pixi_known_envs:-}" ]; then
        # Pixi wraps the names in color codes even when stdout is a pipe, and
        # those codes are part of the captured text unless stripped first.
        _pixi_known_envs="$(pixi workspace environment list 2>/dev/null | sed -n -e 's/\x1b\[[0-9;]*m//g' -e 's/^- \(.*\):$/\1/p')"
    fi
    printf '%s\n' "$_pixi_known_envs" | grep -qxF "$name"
}
