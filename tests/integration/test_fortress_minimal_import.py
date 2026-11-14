from __future__ import annotations

import subprocess
import sys


def test_role_worker_import_does_not_require_full_sdk_dependencies() -> None:
    code = """
import builtins
blocked = {'fastapi', 'httpx', 'PIL', 'diffusers', 'transformers', 'asyncssh', 'xurcroh_quic'}
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in blocked:
        raise ImportError(f'blocked optional dependency: {name}')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
import xurcroh.fortress.worker_cli
import xurcroh.runtime.stage_sessions
"""
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
