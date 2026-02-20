"""
Catchment network module for pyrrm.

Provides tools for multi-catchment, upstream-to-downstream calibration:

- CatchmentNode, NetworkLink: Network building blocks
- CatchmentNetwork: DAG representation with topological sort and wavefronts
- NetworkDataLoader: Multi-catchment data loading and validation
- CatchmentNetworkRunner: Sequential/parallel upstream-to-downstream calibration
- NetworkCalibrationResult: Aggregated network results
- Visualization: Mermaid diagrams, styled DataFrames, matplotlib plots

Example:
    >>> from pyrrm.network import CatchmentNetwork, CatchmentNetworkRunner
    >>> network = CatchmentNetwork.from_csv('topology.csv', 'links.csv', data_dir='./data')
    >>> runner = CatchmentNetworkRunner(network, default_model_class='GR4J')
    >>> result = runner.run()
"""

from pyrrm.network.topology import (
    NodeCalibrationConfig,
    CatchmentNode,
    NetworkLink,
    CatchmentNetwork,
)

from pyrrm.network.data import (
    CatchmentData,
    CatchmentDataSummary,
    JunctionOverlapInfo,
    DataValidationReport,
    NetworkDataLoader,
)

from pyrrm.network.runner import (
    ResolvedNodeConfig,
    NetworkCalibrationResult,
    CatchmentNetworkRunner,
)

from pyrrm.network.visualization import (
    network_to_mermaid,
    display_network,
    plot_network,
    config_summary,
    link_config_summary,
    data_summary,
    result_to_styled_dataframe,
    result_link_styled,
    result_to_mermaid,
    display_result,
    plot_network_hydrographs,
    plot_network_fdc,
    validation_to_styled_dataframe,
    junction_overlap_styled,
)

__all__ = [
    # Topology
    'NodeCalibrationConfig',
    'CatchmentNode',
    'NetworkLink',
    'CatchmentNetwork',
    # Data
    'CatchmentData',
    'CatchmentDataSummary',
    'JunctionOverlapInfo',
    'DataValidationReport',
    'NetworkDataLoader',
    # Runner
    'ResolvedNodeConfig',
    'NetworkCalibrationResult',
    'CatchmentNetworkRunner',
    # Visualization
    'network_to_mermaid',
    'display_network',
    'plot_network',
    'config_summary',
    'link_config_summary',
    'data_summary',
    'result_to_styled_dataframe',
    'result_link_styled',
    'result_to_mermaid',
    'display_result',
    'plot_network_hydrographs',
    'plot_network_fdc',
    'validation_to_styled_dataframe',
    'junction_overlap_styled',
]
