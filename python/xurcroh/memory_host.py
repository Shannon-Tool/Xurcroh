from __future__ import annotations

import ctypes
import mmap
import os
from pathlib import Path

from xurcroh.errors import CompatibilityError

MFD_CLOEXEC = 0x0001
MFD_ALLOW_SEALING = 0x0002
F_ADD_SEALS = 0x0409
F_SEAL_ALL = 0x0001 | 0x0002 | 0x0004 | 0x0008 | 0x0010
PR_SET_DUMPABLE = 4
PR_SET_PTRACER = 17


def protect_process(*, lock_memory: bool = False) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl.argtypes = [
        ctypes.c_int,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
    ]
    if libc.prctl(PR_SET_DUMPABLE, 0, 0, 0, 0) != 0:
        raise CompatibilityError("failed to disable process core dumps")
    libc.prctl(PR_SET_PTRACER, 0, 0, 0, 0)
    if lock_memory:
        libc.mlockall.argtypes = [ctypes.c_int]
        if libc.mlockall(1 | 2) != 0:
            raise CompatibilityError(
                "failed to lock process memory; check CAP_IPC_LOCK and memlock limits"
            )


class SecureMemfd:
    def __init__(self, payload: bytes) -> None:
        self.closed = False
        try:
            self.fd = os.memfd_create("xurcroh_shard", MFD_CLOEXEC | MFD_ALLOW_SEALING)
            os.write(self.fd, payload)
            import fcntl

            fcntl.fcntl(self.fd, F_ADD_SEALS, F_SEAL_ALL)
            self.mapping = mmap.mmap(self.fd, len(payload), access=mmap.ACCESS_READ)
        except (OSError, AttributeError) as error:
            raise CompatibilityError("sealed memfd is unavailable on this platform") from error

    def read(self) -> bytes:
        if self.closed:
            raise RuntimeError("secure memfd is closed")
        return self.mapping[:]

    def __enter__(self) -> SecureMemfd:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        if not self.closed:
            self.mapping.close()
            os.close(self.fd)
            self.closed = True


def secure_stage_path(path: str | Path) -> bytes:
    stage_path = Path(path)
    if stage_path.suffix != ".xurcroh":
        raise CompatibilityError("secure staging accepts only .xurcroh artifacts")
    with SecureMemfd(stage_path.read_bytes()) as secure:
        return secure.read()
