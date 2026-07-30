from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from ray.util.state import list_nodes


class Box:
    def __init__(self, inner):
        self._inner = inner

    @property
    def inner(self):
        return self._inner


def compute_ray_pin_head_options():
    # The state API cannot resolve the head under a ray:// client connection
    # (its helper calls ray.init again); head pinning is an ops nicety, so
    # degrade to default scheduling instead of failing the caller.
    try:
        head_node_id = _get_head_node_id()
    except Exception:
        return {}
    return {
        "scheduling_strategy": NodeAffinitySchedulingStrategy(
            node_id=head_node_id,
            soft=False,
        )
    }


def _get_head_node_id() -> str:
    for node in list_nodes():
        if node.is_head_node:
            return node.node_id
    raise RuntimeError("Could not find a head node in the Ray cluster")
