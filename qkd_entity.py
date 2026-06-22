"""
Basic entities for Satellite-HAP-GS QKD resource allocation.
"""

from dataclasses import dataclass
from enum import Enum


class NodeType(str, Enum):
    GS = "GS"
    SAT = "SAT"
    HAP = "HAP"


class LinkType(str, Enum):
    SAT_GS = "SAT_GS"
    HAP_GS = "HAP_GS"
    SAT_HAP = "SAT_HAP"


@dataclass(frozen=True)
class QKDNode:
    node_id: str
    node_type: NodeType
    lat: float
    lon: float
    altitude_km: float
    connection_limit: int = 1


@dataclass(frozen=True)
class QKDLinkCapacity:
    time_slot: int
    u: str
    v: str
    link_type: LinkType
    capacity_bits: float
    visible: bool = True

    def edge_key(self):
        return tuple(sorted((self.u, self.v)))


@dataclass(frozen=True)
class KeyDemand:
    demand_id: str
    time_slot: int
    src: str
    dst: str
    key_bits: float
