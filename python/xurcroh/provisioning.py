from __future__ import annotations

import logging
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

    logging.getLogger(__name__).debug(
        "SDK provisioning ignores the controller endpoint %r; no controller is wired",
        controller,
    )
    del controller
    return FortressDeployment(cluster, models).start()


def provision_cluster_sync(
    cluster: ClusterConfig,
    models: tuple[tuple[str, ModelDefinition], ...],
    *,
    controller: str,
) -> FortressDeployment:
    from xurcroh.fortress.deployment import FortressDeployment

    return FortressDeployment(cluster, models).start()
