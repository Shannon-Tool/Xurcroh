from __future__ import annotations

import argparse

from Xurcroh import Xurcroh


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local quantized model with Fortress")
    parser.add_argument("checkpoint", help="GGUF file or EXL2/EXL3 checkpoint directory")
    parser.add_argument("--name", default="quantized")
    parser.add_argument("--prompt", default="Explain secret sharing in one sentence.")
    args = parser.parse_args()
    with Xurcroh(model_path={args.name: args.checkpoint}) as client:
        report = client.preflight(args.name)[args.name]
        if not report.supported:
            raise SystemExit("; ".join(report.issues))
        response = client.chat.completions.create(
            model=args.name,
            messages=[{"role": "user", "content": args.prompt}],
            temperature=0,
            max_tokens=64,
        )
        print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
