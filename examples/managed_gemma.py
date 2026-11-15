from __future__ import annotations

import argparse

from Xurcroh import Xurcroh

MODEL_ID = "google/gemma-4-E2B-it"
REVISION = "4adf15748de9e1a0097b6274cee9c15836c41c37"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Gemma 4 E2B through a managed Fortress grid")
    parser.add_argument(
        "--host",
        action="append",
        required=True,
        help="SSH worker hostname or IP; repeat shares^2 times (four by default)",
    )
    parser.add_argument("--ssh-user", default="ubuntu")
    parser.add_argument("--ssh-key", default="~/.ssh/id_ed25519")
    parser.add_argument("--prompt", default="Hello!")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument(
        "--shares",
        type=int,
        default=2,
        help="shares per grid dimension; requires shares^2 worker hosts",
    )
    parser.add_argument("--datagram-tunnel", action="store_true")
    args = parser.parse_args()
    if args.shares < 2:
        parser.error("--shares must be at least 2")
    expected = args.shares**2
    if len(args.host) != expected:
        parser.error(f"this grid requires exactly {expected} --host values")
    cluster = {
        "workers": [
            {
                "name": f"gpu-{index}",
                "host": host,
                "ssh": {"username": args.ssh_user, "key_path": args.ssh_key},
            }
            for index, host in enumerate(args.host)
        ],
        "datagram_tunnel": args.datagram_tunnel,
    }
    with Xurcroh(
        cluster=cluster,
        model_path={
            "gemma": {
                "path": MODEL_ID,
                "revision": REVISION,
                "engine": "fortress",
                "fortress_shares": args.shares,
                "allow_download": True,
            }
        },
    ) as client:
        response = client.chat.completions.create(
            model="gemma",
            messages=[{"role": "user", "content": args.prompt}],
            max_tokens=args.max_tokens,
        )
        print(response.choices[0].message.content or "")


if __name__ == "__main__":
    main()
