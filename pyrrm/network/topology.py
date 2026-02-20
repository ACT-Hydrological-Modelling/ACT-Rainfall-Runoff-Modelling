"""
Catchment network topology: nodes, links, and DAG operations.

Represents a river network as a directed acyclic graph (DAG) of
CatchmentNode objects connected by NetworkLink edges. Provides
topological sorting and wavefront extraction for dependency-aware
parallel scheduling.

Example:
    >>> from pyrrm.network.topology import CatchmentNetwork, CatchmentNode
    >>> nodes = [
    ...     CatchmentNode(id='hw1', name='Upper Creek', area_km2=50, downstream_id='j1'),
    ...     CatchmentNode(id='hw2', name='Side Creek', area_km2=30, downstream_id='j1'),
    ...     CatchmentNode(id='j1', name='Mid River', area_km2=80, downstream_id='outlet'),
    ...     CatchmentNode(id='outlet', name='Lower River', area_km2=120),
    ... ]
    >>> net = CatchmentNetwork(nodes)
    >>> net.topological_order()
    ['hw1', 'hw2', 'j1', 'outlet']
    >>> net.wavefronts()
    [['hw1', 'hw2'], ['j1'], ['outlet']]
"""

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Dict, List, Set, Tuple, Optional, Any, TYPE_CHECKING,
)
import csv
import logging
import warnings

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pyrrm.calibration.objective_functions import ObjectiveFunction

logger = logging.getLogger(__name__)

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class NodeCalibrationConfig:
    """Per-node calibration overrides. Fields left as None inherit the network default."""
    model_class: Optional[str] = None
    model_params: Optional[Dict[str, Any]] = None
    parameter_bounds: Optional[Dict[str, Tuple]] = None
    objective: Optional[Any] = None
    flow_transformation: Optional[Any] = None
    algorithm: Optional[Dict[str, Any]] = None
    warmup_period: Optional[int] = None


@dataclass
class CatchmentNode:
    """A subcatchment in the river network.

    Attributes:
        id: Unique identifier (e.g. gauge number '410734').
        name: Human-readable name.
        area_km2: Local incremental catchment area.
        inputs: P/PET DataFrame (set by data loader or manually).
        observed: Observed flow array at gauge (None if ungauged).
        downstream_id: ID of the downstream node (None = outlet).
        calibration_config: Per-node overrides (None = use network defaults).
        metadata: Arbitrary extra metadata.
    """
    id: str
    name: str = ''
    area_km2: float = 0.0
    inputs: Optional[pd.DataFrame] = field(default=None, repr=False)
    observed: Optional[np.ndarray] = field(default=None, repr=False)
    downstream_id: Optional[str] = None
    calibration_config: Optional[NodeCalibrationConfig] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_gauged(self) -> bool:
        return self.observed is not None and len(self.observed) > 0

    @property
    def is_outlet(self) -> bool:
        return self.downstream_id is None or self.downstream_id == ''


@dataclass
class NetworkLink:
    """Defines the routing on a link FROM upstream_id TO downstream_id.

    Attributes:
        upstream_id: Source node.
        downstream_id: Destination node.
        routing_method: Routing algorithm ('muskingum', 'lag', 'none').
        routing_params: Initial / fixed routing parameter values.
        calibrate_routing: Whether K, m, n_subreaches are calibrated.
        routing_bounds: Custom bounds, e.g. ``{'routing_K': (0.5, 30.0)}``.
    """
    upstream_id: str
    downstream_id: str
    routing_method: str = 'muskingum'
    routing_params: Optional[Dict[str, float]] = field(default_factory=lambda: {
        'routing_K': 5.0, 'routing_m': 0.8, 'routing_n_subreaches': 3,
    })
    calibrate_routing: bool = True
    routing_bounds: Optional[Dict[str, Tuple[float, float]]] = None


# ---------------------------------------------------------------------------
# CatchmentNetwork
# ---------------------------------------------------------------------------

class CatchmentNetwork:
    """Directed acyclic graph (DAG) representing a river network.

    Args:
        nodes: List of CatchmentNode instances.
        links: Optional list of NetworkLink instances defining routing
            between connected nodes. If None, default links are created
            for every (node, downstream) pair with ``routing_method='none'``.
    """

    def __init__(
        self,
        nodes: List[CatchmentNode],
        links: Optional[List[NetworkLink]] = None,
    ):
        self._nodes: Dict[str, CatchmentNode] = {n.id: n for n in nodes}
        self._links: Dict[Tuple[str, str], NetworkLink] = {}

        if links:
            for lnk in links:
                self._links[(lnk.upstream_id, lnk.downstream_id)] = lnk

        self._children: Dict[str, List[str]] = {nid: [] for nid in self._nodes}
        self._parents: Dict[str, List[str]] = {nid: [] for nid in self._nodes}

        for node in self._nodes.values():
            ds = node.downstream_id
            if ds and ds in self._nodes:
                self._children[node.id].append(ds)
                self._parents[ds].append(node.id)

        self._validate_structure()

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    @property
    def nodes(self) -> Dict[str, CatchmentNode]:
        return dict(self._nodes)

    @property
    def node_ids(self) -> List[str]:
        return list(self._nodes.keys())

    @property
    def links(self) -> Dict[Tuple[str, str], NetworkLink]:
        return dict(self._links)

    def get_node(self, node_id: str) -> CatchmentNode:
        if node_id not in self._nodes:
            raise KeyError(f"Node '{node_id}' not in network")
        return self._nodes[node_id]

    def get_link(self, upstream_id: str, downstream_id: str) -> Optional[NetworkLink]:
        return self._links.get((upstream_id, downstream_id))

    def get_incoming_links(self, node_id: str) -> List[NetworkLink]:
        return [
            self._links[(uid, node_id)]
            for uid in self._parents.get(node_id, [])
            if (uid, node_id) in self._links
        ]

    def upstream_ids(self, node_id: str) -> List[str]:
        """Direct upstream (parent) node IDs."""
        return list(self._parents.get(node_id, []))

    def all_upstream_ids(self, node_id: str) -> Set[str]:
        """All upstream node IDs (transitive closure via BFS)."""
        visited: Set[str] = set()
        queue = deque(self._parents.get(node_id, []))
        while queue:
            uid = queue.popleft()
            if uid not in visited:
                visited.add(uid)
                queue.extend(self._parents.get(uid, []))
        return visited

    @property
    def headwater_ids(self) -> List[str]:
        return [nid for nid, parents in self._parents.items() if not parents]

    @property
    def outlet_ids(self) -> List[str]:
        return [nid for nid, node in self._nodes.items() if node.is_outlet]

    # ------------------------------------------------------------------
    # DAG operations
    # ------------------------------------------------------------------
    def topological_order(self) -> List[str]:
        """Return node IDs upstream-to-downstream (Kahn's algorithm)."""
        in_degree = {nid: len(self._parents[nid]) for nid in self._nodes}
        queue = deque(nid for nid, d in in_degree.items() if d == 0)
        order = []
        while queue:
            nid = queue.popleft()
            order.append(nid)
            node = self._nodes[nid]
            ds = node.downstream_id
            if ds and ds in self._nodes:
                in_degree[ds] -= 1
                if in_degree[ds] == 0:
                    queue.append(ds)
        if len(order) != len(self._nodes):
            raise RuntimeError(
                "Cycle detected in network (topological sort incomplete). "
                f"Sorted {len(order)}/{len(self._nodes)} nodes."
            )
        return order

    def wavefronts(self) -> List[List[str]]:
        """Group node IDs by dependency level for parallel scheduling.

        Wavefront 0: headwater nodes (no upstream dependencies).
        Wavefront k: nodes whose upstream nodes are all in wavefronts 0..k-1.
        """
        in_degree = {nid: len(self._parents[nid]) for nid in self._nodes}
        current = [nid for nid, d in in_degree.items() if d == 0]
        result = []
        remaining = dict(in_degree)

        while current:
            result.append(sorted(current))
            next_front = []
            for nid in current:
                node = self._nodes[nid]
                ds = node.downstream_id
                if ds and ds in remaining:
                    remaining[ds] -= 1
                    if remaining[ds] == 0:
                        next_front.append(ds)
                del remaining[nid]
            current = next_front

        return result

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def _validate_structure(self) -> None:
        errors = self.validate()
        if errors:
            for e in errors:
                logger.warning("Network validation: %s", e)

    def validate(self) -> List[str]:
        """Check structural integrity: cycles, missing references, etc."""
        issues = []
        for node in self._nodes.values():
            ds = node.downstream_id
            if ds and ds not in self._nodes:
                issues.append(
                    f"Node '{node.id}' references downstream '{ds}' "
                    f"which is not in the network"
                )
        try:
            self.topological_order()
        except RuntimeError as e:
            issues.append(str(e))

        if not self.outlet_ids:
            issues.append("No outlet node found (all nodes have a downstream_id)")

        return issues

    # ------------------------------------------------------------------
    # CSV / YAML factories
    # ------------------------------------------------------------------
    @classmethod
    def from_csv(
        cls,
        topology_path: str,
        links_path: Optional[str] = None,
        data_dir: Optional[str] = None,
        load_data: bool = True,
        **loader_kwargs,
    ) -> 'CatchmentNetwork':
        """Build a CatchmentNetwork from CSV files.

        Args:
            topology_path: Path to topology CSV. Required columns:
                ``id, downstream_id, area_km2``. Optional: ``name``,
                ``input_file`` / ``precip_file, pet_file, observed_file``.
            links_path: Path to link routing CSV. Columns:
                ``upstream_id, downstream_id, routing_method, K_init, ...``
            data_dir: Base directory for resolving relative data file paths.
            load_data: If True, load catchment data via NetworkDataLoader.
            **loader_kwargs: Extra kwargs for NetworkDataLoader.

        Returns:
            CatchmentNetwork instance with data loaded (if requested).
        """
        topo_df = pd.read_csv(topology_path)
        topo_df.columns = [c.strip().lower() for c in topo_df.columns]

        if 'id' not in topo_df.columns:
            raise ValueError("Topology CSV must have an 'id' column")

        nodes = []
        for _, row in topo_df.iterrows():
            ds = row.get('downstream_id', None)
            if pd.isna(ds) or ds == '':
                ds = None
            else:
                ds = str(ds).strip()

            cal_cfg = None
            if 'model' in row and pd.notna(row.get('model')):
                cal_cfg = NodeCalibrationConfig(model_class=str(row['model']))

            nodes.append(CatchmentNode(
                id=str(row['id']).strip(),
                name=str(row.get('name', '')).strip(),
                area_km2=float(row.get('area_km2', 0)),
                downstream_id=ds,
                calibration_config=cal_cfg,
                metadata={
                    k: row[k] for k in row.index
                    if k not in ('id', 'name', 'downstream_id', 'area_km2', 'model')
                    and pd.notna(row[k])
                },
            ))

        links = None
        if links_path:
            links = cls._parse_links_csv(links_path)

        network = cls(nodes, links)

        if load_data and data_dir is not None:
            from pyrrm.network.data import NetworkDataLoader
            loader = NetworkDataLoader(
                topology_csv=topology_path,
                data_dir=data_dir,
                **loader_kwargs,
            )
            catchment_data = loader.load()
            report = loader.validate()
            if not report.is_valid():
                raise ValueError(f"Data validation failed:\n{report}")
            for w in report.warnings:
                warnings.warn(w)
            for node_id, cdata in catchment_data.items():
                if node_id in network._nodes:
                    network._nodes[node_id].inputs = cdata.inputs
                    network._nodes[node_id].observed = cdata.observed
                    network._nodes[node_id].area_km2 = cdata.area_km2

        return network

    @staticmethod
    def _parse_links_csv(path: str) -> List[NetworkLink]:
        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]
        links = []
        for _, row in df.iterrows():
            params = {}
            bounds = {}
            for col in ('k_init', 'routing_k', 'k'):
                if col in row and pd.notna(row[col]):
                    params['routing_K'] = float(row[col])
                    break
            for col in ('m_init', 'routing_m', 'm'):
                if col in row and pd.notna(row[col]):
                    params['routing_m'] = float(row[col])
                    break
            for col in ('n_subreaches_init', 'n_subreaches', 'routing_n_subreaches'):
                if col in row and pd.notna(row[col]):
                    params['routing_n_subreaches'] = int(row[col])
                    break

            if 'k_min' in row and pd.notna(row.get('k_min')):
                bounds['routing_K'] = (float(row['k_min']), float(row.get('k_max', 200)))
            if 'm_min' in row and pd.notna(row.get('m_min')):
                bounds['routing_m'] = (float(row['m_min']), float(row.get('m_max', 1.5)))

            cal = True
            if 'calibrate_routing' in row:
                val = row['calibrate_routing']
                if isinstance(val, str):
                    cal = val.strip().lower() in ('true', '1', 'yes')
                else:
                    cal = bool(val)

            links.append(NetworkLink(
                upstream_id=str(row['upstream_id']).strip(),
                downstream_id=str(row['downstream_id']).strip(),
                routing_method=str(row.get('routing_method', 'muskingum')).strip(),
                routing_params=params or None,
                calibrate_routing=cal,
                routing_bounds=bounds or None,
            ))
        return links

    @classmethod
    def from_yaml(cls, path: str, data_dir: Optional[str] = None,
                  load_data: bool = True, **loader_kwargs) -> 'CatchmentNetwork':
        """Build a CatchmentNetwork from a YAML configuration file.

        See the plan document for the expected YAML schema.
        """
        if not YAML_AVAILABLE:
            raise ImportError(
                "PyYAML is required to load YAML network configs. "
                "Install with: pip install pyyaml"
            )
        with open(path) as f:
            cfg = yaml.safe_load(f)

        nodes = []
        for nid, ncfg in cfg.get('nodes', {}).items():
            cal_cfg = None
            if 'calibration' in ncfg:
                cc = ncfg['calibration']
                cal_cfg = NodeCalibrationConfig(
                    model_class=cc.get('model'),
                    objective=cc.get('objective'),
                    algorithm=cc.get('algorithm'),
                    warmup_period=cc.get('warmup_days'),
                )
            nodes.append(CatchmentNode(
                id=str(nid),
                name=ncfg.get('name', ''),
                area_km2=ncfg.get('area_km2', 0),
                downstream_id=ncfg.get('downstream_id'),
                calibration_config=cal_cfg,
                metadata={
                    k: v for k, v in ncfg.items()
                    if k not in ('name', 'area_km2', 'downstream_id', 'calibration')
                },
            ))

        links = None
        if 'links' in cfg:
            links = []
            for lcfg in cfg['links']:
                params = {}
                if 'K' in lcfg:
                    params['routing_K'] = float(lcfg['K'])
                if 'm' in lcfg:
                    params['routing_m'] = float(lcfg['m'])
                if 'n_subreaches' in lcfg:
                    params['routing_n_subreaches'] = int(lcfg['n_subreaches'])
                links.append(NetworkLink(
                    upstream_id=str(lcfg.get('upstream', lcfg.get('upstream_id'))),
                    downstream_id=str(lcfg.get('downstream', lcfg.get('downstream_id'))),
                    routing_method=lcfg.get('routing', 'muskingum'),
                    routing_params=params or None,
                    calibrate_routing=lcfg.get('calibrate_routing', True),
                ))

        return cls(nodes, links)

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        n = len(self._nodes)
        l = len(self._links)
        hw = len(self.headwater_ids)
        out = len(self.outlet_ids)
        return (
            f"CatchmentNetwork({n} nodes, {l} links, "
            f"{hw} headwaters, {out} outlets)"
        )

    def summary(self) -> str:
        """Human-readable text summary."""
        lines = [repr(self), '']
        for wf_idx, wf in enumerate(self.wavefronts()):
            lines.append(f"  Wavefront {wf_idx}: {wf}")
        lines.append('')
        lines.append("Nodes:")
        for nid in self.topological_order():
            node = self._nodes[nid]
            gauged = 'gauged' if node.is_gauged else 'ungauged'
            lines.append(
                f"  {nid} ({node.name}) -- {node.area_km2:.0f} km², "
                f"{gauged}, ds={node.downstream_id}"
            )
        if self._links:
            lines.append("\nLinks:")
            for (uid, did), lnk in self._links.items():
                cal = 'calibrate' if lnk.calibrate_routing else 'fixed'
                lines.append(
                    f"  {uid} -> {did}: {lnk.routing_method} ({cal})"
                )
        return '\n'.join(lines)
