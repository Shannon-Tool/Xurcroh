from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, NoEncryption, PrivateFormat

from xurcroh.errors import ShardIntegrityError
from xurcroh.shards.format import read_shard
from xurcroh.shards.recipients import open_stage_key
from xurcroh.stage_worker import DenseStage


def main() -> None:
    manifest = Path(sys.argv[1])
    identity_path = Path(sys.argv[2])
    private = X25519PrivateKey.from_private_bytes(identity_path.read_bytes())
    stage = DenseStage(manifest, 0, private)
    document = json.loads(manifest.read_text())
    boundary_opened = False
    try:
        read_shard(
            manifest.parent / document["boundary"]["file"],
            private.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption()),
        )
        boundary_opened = True
    except ShardIntegrityError:
        pass
    peer_opened = False
    try:
        envelope = json.loads((manifest.parent / document["stages"][1]["key_envelope"]).read_text())
        open_stage_key(
            envelope,
            private,
            manifest_digest=document["digest"],
            checkpoint_fingerprint=document["checkpoint_fingerprint"],
        )
        peer_opened = True
    except ShardIntegrityError:
        pass
    result = {
        "master_env_present": "XURCROH_MASTER_KEY" in os.environ,
        "boundary_opened_with_private_identity": boundary_opened,
        "peer_opened_with_own_identity": peer_opened,
        "basis_tensor_names": sorted(
            name for name in stage._tensors if "basis" in name or "transition" in name
        ),
    }
    print(json.dumps(result), flush=True)
    time.sleep(30)
    stage.close()


if __name__ == "__main__":
    main()
