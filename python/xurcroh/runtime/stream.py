from __future__ import annotations

from collections.abc import Callable, Iterator


class ManagedStream[T]:
    def __init__(self, iterator: Iterator[T], cleanup: Callable[[], None]) -> None:
        self._iterator = iterator
        self._cleanup = cleanup
        self._closed = False

    def __iter__(self) -> ManagedStream[T]:
        return self

    def __next__(self) -> T:
        if self._closed:
            raise StopIteration
        try:
            return next(self._iterator)
        except StopIteration:
            self.close()
            raise
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        close = getattr(self._iterator, "close", None)
        try:
            if close is not None:
                close()
        finally:
            self._cleanup()

    def __del__(self) -> None:
        self.close()
