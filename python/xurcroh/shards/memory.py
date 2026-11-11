from __future__ import annotations

import ctypes
import gc
import mmap

from xurcroh.errors import ShardIntegrityError


def _address(buffer: bytearray | mmap.mmap) -> int:
    view = (ctypes.c_char * len(buffer)).from_buffer(buffer)
    return ctypes.addressof(view)


def secure_zero(buffer: bytearray | mmap.mmap) -> None:
    if buffer:
        ctypes.memset(_address(buffer), 0, len(buffer))


def lock_buffer(buffer: bytearray | mmap.mmap) -> None:
    if not buffer:
        return
    libc = ctypes.CDLL(None, use_errno=True)
    if not hasattr(libc, "mlock"):
        raise ShardIntegrityError("platform does not provide mlock for secure shard decryption")
    libc.mlock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    libc.mlock.restype = ctypes.c_int
    if libc.mlock(_address(buffer), len(buffer)) != 0:
        error_number = ctypes.get_errno()
        raise ShardIntegrityError(
            f"cannot lock decrypted shard pages in RAM (errno {error_number}); "
            "raise the worker memlock limit or use a smaller stage"
        )


def unlock_buffer(buffer: bytearray | mmap.mmap) -> None:
    if buffer:
        libc = ctypes.CDLL(None, use_errno=True)
        if hasattr(libc, "munlock"):
            libc.munlock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            libc.munlock.restype = ctypes.c_int
            libc.munlock(_address(buffer), len(buffer))


def release_host_memory() -> None:
    gc.collect()
    libc = ctypes.CDLL(None)
    if hasattr(libc, "malloc_trim"):
        libc.malloc_trim(0)
