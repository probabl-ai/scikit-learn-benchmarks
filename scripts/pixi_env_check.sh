#!/usr/bin/env bash
# Shared helper, sourced by run.sh and dispatch.sh: checks whether a name is
# a pixi environment actually defined in this repo's pixi.toml, so a typo'd
# or removed env name fails fast with a clear message instead of failing
# later (after e.g. a slow sklearn ref checkout, or a CI dispatch).

pixi_env_exists() {
    local name="$1"
    if [ -z "${_pixi_known_envs:-}" ]; then
        _pixi_known_envs="$(pixi workspace environment list 2>/dev/null | sed -n 's/^- \(.*\):$/\1/p')"
    fi
    printf '%s\n' "$_pixi_known_envs" | grep -qxF "$name"
}
