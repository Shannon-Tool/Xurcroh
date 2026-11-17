#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate public synthetic EXL decoder fixtures")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    from tests.formats.exl3_utils import exl3_fixture
    from tests.formats.exl_utils import exl2_fixture, gptq_fixture

    exl2 = args.output / "exl2"
    gptq = args.output / "gptq"
    exl3 = args.output / "exl3"
    exl2.mkdir(parents=True, exist_ok=True)
    gptq.mkdir(parents=True, exist_ok=True)
    exl3.mkdir(parents=True, exist_ok=True)
    exl2_fixture(exl2)
    gptq_fixture(gptq)
    exl3_fixture(exl3, 2, "3inst")


if __name__ == "__main__":
    main()
