# Copyright (C) 2026, contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
"""Serve-profile routing.

Resolves the ``vllm serve`` parameter set for a model directory from a
declarative profile table instead of per-model hand-written launch commands.
The table is shared with the aiter-side router (layers 2/3) and lives at
``<aiter>/configs/runtime_routing.yaml`` by default, overridable via
``AITER_RUNTIME_ROUTING``; ``AITER_ROUTING_SERVE=0`` force-disables this
layer (resolver then returns an explicit ``routed: false`` result and callers
fall back to their own defaults).

Usage (standalone, from a container entrypoint or shell):

    python -m vllm.runtime_routing /opt/models/<model_dir> [--format shell|json]

``--format shell`` prints ``export K=V`` lines followed by one ``vllm-args``
line (shell-quoted) suitable for ``eval``; ``--format json`` prints the raw
resolution dict.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_ROUTING_ENV = "AITER_RUNTIME_ROUTING"
_SERVE_SWITCH = "AITER_ROUTING_SERVE"


@dataclass
class ServeResolution:
    profile: str
    params: dict[str, Any] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)
    unsupported: bool = False
    routed: bool = True

    def to_json(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "params": self.params,
            "env": self.env,
            "unsupported": self.unsupported,
            "routed": self.routed,
        }


def _table_path() -> str:
    path = os.environ.get(_ROUTING_ENV, "").strip()
    if path:
        return path
    # Default: table ships inside the aiter checkout/configs. Search common
    # locations so this works in the image layout (/src/aiter) and in
    # editable installs.
    candidates = [
        Path(__file__).resolve().parent.parent / "configs" / "runtime_routing.yaml",
        Path("/src/aiter/configs/runtime_routing.yaml"),
        Path("/opt/aiter/configs/runtime_routing.yaml"),
    ]
    for c in candidates:
        if c.is_file():
            return str(c)
    return str(candidates[0])


@lru_cache(maxsize=1)
def _profiles() -> dict[str, dict[str, Any]]:
    try:
        import yaml

        with open(_table_path()) as f:
            table = yaml.safe_load(f) or {}
        profiles = table.get("serve_profiles") or {}
        if not isinstance(profiles, dict):
            raise ValueError("serve_profiles is not a mapping")
        return profiles
    except Exception as e:  # noqa: BLE001 - absent table means "not routed"
        logger.warning("serve routing table unusable (%s); layer-1 disabled", e)
        return {}


def _dig(cfg: dict[str, Any], dotted: str) -> Any:
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _match(match_rules: dict[str, Any], model_cfg: dict[str, Any]) -> bool:
    for dotted, cond in match_rules.items():
        actual = _dig(model_cfg, dotted)
        if isinstance(cond, list):
            # cond list = membership; actual may itself be a list (e.g.
            # architectures) -> any-intersection, else scalar containment.
            if isinstance(actual, list):
                if not set(actual) & set(cond):
                    return False
            elif actual not in cond:
                return False
        elif actual != cond:
            return False
    return True


def resolve_serve_params(model_dir: str) -> ServeResolution:
    """Match ``model_dir/config.json`` against the profile table (first hit wins)."""
    if os.environ.get(_SERVE_SWITCH, "1").strip() in ("0", "false", "False"):
        return ServeResolution(profile="<disabled>", routed=False)
    cfg_path = Path(model_dir) / "config.json"
    try:
        with open(cfg_path) as f:
            model_cfg = json.load(f)
    except Exception as e:  # noqa: BLE001
        logger.warning("cannot read %s (%s); layer-1 disabled", cfg_path, e)
        return ServeResolution(profile="<unreadable>", routed=False)

    for name, profile in _profiles().items():
        if _match(profile.get("match", {}), model_cfg):
            params = dict(profile.get("params") or {})
            env = {str(k): str(v) for k, v in (params.pop("env", {}) or {}).items()}
            unsupported = bool(params.pop("unsupported", False))
            return ServeResolution(
                profile=name, params=params, env=env, unsupported=unsupported
            )
    logger.warning("no serve profile matched %s; layer-1 disabled", model_dir)
    return ServeResolution(profile="<no-match>", routed=False)


def _param_to_args(params: dict[str, Any]) -> list[str]:
    """Render resolved params as vllm-serve-style CLI arguments."""
    args: list[str] = []
    spec = params.get("speculative")
    for key, value in params.items():
        if key == "speculative":
            continue
        if isinstance(value, bool):
            args.append(f"--{key.replace('_', '-')}" if value else f"--no-{key.replace('_', '-')}")
        elif isinstance(value, (int, float, str)):
            args += [f"--{key.replace('_', '-')}", str(value)]
    if isinstance(spec, dict):
        payload = {
            k: v
            for k, v in spec.items()
            if k not in ("draft_model_match", "model")  # model filled by caller/tooling
        }
        args += ["--speculative-config", json.dumps(payload)]
    return args


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("--format", choices=("shell", "json"), default="shell")
    opts = parser.parse_args()
    res = resolve_serve_params(opts.model_dir)
    if opts.format == "json":
        print(json.dumps(res.to_json(), indent=2))
        return
    if res.routed and not res.unsupported:
        for k, v in res.env.items():
            print(f"export {k}={shlex.quote(v)}")
        print("vllm-args " + " ".join(shlex.quote(a) for a in _param_to_args(res.params)))
    else:
        print(f"# serve routing not applied: profile={res.profile}")
        raise SystemExit(2)


if __name__ == "__main__":
    _main()
