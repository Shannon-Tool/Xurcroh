from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from xurcroh.config import ClusterConfig, ModelDefinition

if TYPE_CHECKING:
    from xurcroh.fortress.deployment import FortressDeployment


async def provision_cluster(
    cluster: ClusterConfig,
    models: tuple[tuple[str, ModelDefinition], ...],
    *,
    controller: str,
) -> FortressDeployment:
    from xurcroh.fortress.deployment import FortressDeployment

    del controller
    return FortressDeployment(cluster, models).start()


def provision_cluster_sync(
    cluster: ClusterConfig,
    models: tuple[tuple[str, ModelDefinition], ...],
    *,
    controller: str,
) -> FortressDeployment:
    from xurcroh.fortress.deployment import FortressDeployment

    failure: list[BaseException] = []
    result: list[FortressDeployment] = []

    def run() -> None:
        try:
            result.append(FortressDeployment(cluster, models).start())
        except BaseException as error:
            failure.append(error)

    thread = threading.Thread(target=run, name="xurcroh-provision")
    thread.start()
    thread.join()
    if failure:
        raise failure[0]
    return result[0]
