"""
Multi-catchment data loading, validation, and alignment for network calibration.

Handles two input layouts:
  A. Single file per catchment (date, rainfall, pet, observed_flow)
  B. Separate files per variable (precip_file, pet_file, observed_file)

Each catchment retains its FULL available record -- no global trimming.
Temporal alignment for objective evaluation happens at junction aggregation
time in the CatchmentNetworkRunner, not here.

Example:
    >>> from pyrrm.network.data import NetworkDataLoader
    >>> loader = NetworkDataLoader('topology.csv', data_dir='./data/')
    >>> data = loader.load()
    >>> report = loader.validate()
    >>> print(report)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
import warnings

import numpy as np
import pandas as pd

from pyrrm.data import COLUMN_ALIASES, resolve_column

logger = logging.getLogger(__name__)

# Backward-compatible aliases derived from the canonical source
PRECIP_SYNONYMS = COLUMN_ALIASES["precipitation"]
PET_SYNONYMS = COLUMN_ALIASES["pet"]
FLOW_SYNONYMS = COLUMN_ALIASES["observed_flow"]
DATE_SYNONYMS = COLUMN_ALIASES["date"]


def _find_column(df: pd.DataFrame, synonyms: List[str]) -> Optional[str]:
    """Case-insensitive column lookup (delegates to resolve_column when possible)."""
    # Map synonym lists to canonical names for direct delegation
    _list_to_canonical = {
        id(PRECIP_SYNONYMS): "precipitation",
        id(PET_SYNONYMS): "pet",
        id(FLOW_SYNONYMS): "observed_flow",
        id(DATE_SYNONYMS): "date",
    }
    canonical = _list_to_canonical.get(id(synonyms))
    if canonical is not None:
        return resolve_column(df, canonical)
    # Fallback for ad-hoc synonym lists
    lower_cols = {c.lower(): c for c in df.columns}
    for s in synonyms:
        if s.lower() in lower_cols:
            return lower_cols[s.lower()]
    return None


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class CatchmentData:
    """Validated data for a single catchment node -- retains its FULL available record."""
    node_id: str
    inputs: pd.DataFrame
    observed: Optional[np.ndarray]
    dates: pd.DatetimeIndex
    area_km2: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CatchmentDataSummary:
    """Per-catchment data quality summary."""
    node_id: str
    n_records: int
    period: Tuple[str, str]
    precip_missing_pct: float
    pet_missing_pct: float
    flow_missing_pct: Optional[float]
    longest_gap_days: int
    source_files: List[str] = field(default_factory=list)


@dataclass
class JunctionOverlapInfo:
    """Analysis of date overlap at a junction node."""
    junction_id: str
    junction_observed_period: Optional[Tuple[str, str]]
    upstream_periods: Dict[str, Tuple[str, str]]
    effective_calibration_period: Optional[Tuple[str, str]]
    effective_days: int
    is_sufficient: bool


@dataclass
class DataValidationReport:
    """Summary of data quality across the network."""
    n_catchments: int
    per_catchment: Dict[str, CatchmentDataSummary] = field(default_factory=dict)
    junction_overlaps: Dict[str, JunctionOverlapInfo] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def is_valid(self) -> bool:
        return len(self.errors) == 0

    def __repr__(self) -> str:
        lines = [
            f"DataValidationReport: {self.n_catchments} catchments, "
            f"{len(self.errors)} errors, {len(self.warnings)} warnings",
        ]
        if self.errors:
            lines.append("\nErrors:")
            for e in self.errors:
                lines.append(f"  - {e}")
        if self.warnings:
            lines.append("\nWarnings:")
            for w in self.warnings[:20]:
                lines.append(f"  - {w}")
            if len(self.warnings) > 20:
                lines.append(f"  ... and {len(self.warnings) - 20} more")
        lines.append("\nPer-Catchment Summary:")
        lines.append(f"  {'ID':<15} {'Records':>8} {'Period':<25} {'P_miss%':>8} {'PET_miss%':>9} {'Q_miss%':>8}")
        lines.append(f"  {'-'*15} {'-'*8} {'-'*25} {'-'*8} {'-'*9} {'-'*8}")
        for nid, s in self.per_catchment.items():
            q_miss = f"{s.flow_missing_pct:.1f}" if s.flow_missing_pct is not None else 'N/A'
            lines.append(
                f"  {nid:<15} {s.n_records:>8} "
                f"{s.period[0]} to {s.period[1]:<10} "
                f"{s.precip_missing_pct:>7.1f} {s.pet_missing_pct:>9.1f} {q_miss:>8}"
            )
        if self.junction_overlaps:
            lines.append("\nJunction Overlaps:")
            for jid, jo in self.junction_overlaps.items():
                suf = '' if jo.is_sufficient else ' (INSUFFICIENT)'
                if jo.effective_calibration_period:
                    lines.append(
                        f"  {jid}: {jo.effective_calibration_period[0]} to "
                        f"{jo.effective_calibration_period[1]} "
                        f"({jo.effective_days} days){suf}"
                    )
                else:
                    lines.append(f"  {jid}: no effective overlap{suf}")
        return '\n'.join(lines)


# ---------------------------------------------------------------------------
# NetworkDataLoader
# ---------------------------------------------------------------------------

class NetworkDataLoader:
    """Load, validate, and prepare data for all catchments in a network.

    Args:
        topology_csv: Path to topology CSV containing node definitions
            and data file references.
        data_dir: Base directory for resolving relative data file paths.
        missing_values: Sentinel values to replace with NaN.
        missing_strategy: How to handle gaps -- 'warn', 'interpolate', or 'drop'.
        date_column: Name of the date column in data files.
        date_format: strptime format for date parsing (None = auto).
        target_frequency: Expected frequency ('D' for daily).
        min_calibration_years: Minimum junction overlap for is_sufficient flag.
    """

    def __init__(
        self,
        topology_csv: str,
        data_dir: Optional[str] = None,
        missing_values: Optional[List[Any]] = None,
        missing_strategy: str = 'warn',
        date_column: str = 'date',
        date_format: Optional[str] = None,
        target_frequency: str = 'D',
        min_calibration_years: float = 2.0,
    ):
        self.topology_csv = Path(topology_csv)
        self.data_dir = Path(data_dir) if data_dir else self.topology_csv.parent
        self.missing_values = missing_values or [-9999, -999, -99.99, -9999.0, -999.0]
        self.missing_strategy = missing_strategy
        self.date_column = date_column
        self.date_format = date_format
        self.target_frequency = target_frequency
        self.min_calibration_years = min_calibration_years

        self._topo_df: Optional[pd.DataFrame] = None
        self._data: Optional[Dict[str, CatchmentData]] = None
        self._report: Optional[DataValidationReport] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def load(self) -> Dict[str, CatchmentData]:
        """Load all catchment data files and return validated data per node."""
        self._topo_df = pd.read_csv(self.topology_csv)
        self._topo_df.columns = [c.strip().lower() for c in self._topo_df.columns]

        data: Dict[str, CatchmentData] = {}
        for _, row in self._topo_df.iterrows():
            node_id = str(row['id']).strip()
            try:
                cdata = self._load_single_node(node_id, row)
                data[node_id] = cdata
            except Exception as e:
                logger.error("Failed to load data for node %s: %s", node_id, e)
                raise

        self._data = data
        return data

    def validate(self) -> DataValidationReport:
        """Run all validation checks and return a report."""
        if self._data is None:
            self.load()

        assert self._data is not None and self._topo_df is not None

        report = DataValidationReport(n_catchments=len(self._data))

        for node_id, cdata in self._data.items():
            summary = self._summarise_catchment(cdata)
            report.per_catchment[node_id] = summary

            if summary.precip_missing_pct > 5:
                report.warnings.append(
                    f"Node {node_id}: {summary.precip_missing_pct:.1f}% precipitation missing"
                )
            if summary.pet_missing_pct > 5:
                report.warnings.append(
                    f"Node {node_id}: {summary.pet_missing_pct:.1f}% PET missing"
                )
            if summary.flow_missing_pct is not None and summary.flow_missing_pct > 5:
                report.warnings.append(
                    f"Node {node_id}: {summary.flow_missing_pct:.1f}% flow missing"
                )

        freq_set = set()
        for cdata in self._data.values():
            if len(cdata.inputs) > 1:
                inferred = pd.infer_freq(cdata.inputs.index[:50])
                if inferred:
                    freq_set.add(inferred)
        if len(freq_set) > 1:
            report.errors.append(
                f"Mixed data frequencies detected: {freq_set}. "
                f"All catchments must use the same frequency."
            )

        junction_overlaps = self._compute_junction_overlaps()
        report.junction_overlaps = junction_overlaps
        for jid, jo in junction_overlaps.items():
            if not jo.is_sufficient:
                report.warnings.append(
                    f"Junction {jid}: effective calibration period "
                    f"({jo.effective_days} days) is less than "
                    f"{self.min_calibration_years} years"
                )

        self._apply_missing_strategy()

        self._report = report
        return report

    def summary(self) -> str:
        """Return a human-readable summary string."""
        if self._report is None:
            self.validate()
        return repr(self._report)

    # ------------------------------------------------------------------
    # Internal: load a single node
    # ------------------------------------------------------------------
    def _load_single_node(self, node_id: str, row: pd.Series) -> CatchmentData:
        """Load data for one node, handling both layout A and B."""

        has_single_file = 'input_file' in row and pd.notna(row.get('input_file'))
        has_separate_files = any(
            col in row and pd.notna(row.get(col))
            for col in ('precip_file', 'pet_file')
        )

        source_files = []
        inputs_df: Optional[pd.DataFrame] = None
        observed_arr: Optional[np.ndarray] = None

        if has_single_file:
            fpath = self._resolve_path(row['input_file'])
            source_files.append(str(fpath))
            df = self._read_csv_with_dates(fpath)
            inputs_df, observed_arr = self._extract_columns(df)

        elif has_separate_files:
            dfs_to_join = []

            if 'precip_file' in row and pd.notna(row.get('precip_file')):
                fp = self._resolve_path(row['precip_file'])
                source_files.append(str(fp))
                pdf = self._read_csv_with_dates(fp)
                pcol = _find_column(pdf, PRECIP_SYNONYMS)
                if pcol:
                    dfs_to_join.append(pdf[[pcol]].rename(columns={pcol: 'precipitation'}))

            if 'pet_file' in row and pd.notna(row.get('pet_file')):
                fp = self._resolve_path(row['pet_file'])
                source_files.append(str(fp))
                edf = self._read_csv_with_dates(fp)
                ecol = _find_column(edf, PET_SYNONYMS)
                if ecol:
                    dfs_to_join.append(edf[[ecol]].rename(columns={ecol: 'pet'}))

            if dfs_to_join:
                inputs_df = dfs_to_join[0]
                for extra in dfs_to_join[1:]:
                    inputs_df = inputs_df.join(extra, how='inner')

            obs_col_name = 'observed_file'
            if obs_col_name in row and pd.notna(row.get(obs_col_name)):
                fp = self._resolve_path(row[obs_col_name])
                source_files.append(str(fp))
                odf = self._read_csv_with_dates(fp)
                fcol = _find_column(odf, FLOW_SYNONYMS)
                if fcol:
                    observed_arr = odf[fcol].values
                    if inputs_df is not None:
                        obs_series = odf[fcol].reindex(inputs_df.index)
                        observed_arr = obs_series.values
        else:
            raise ValueError(
                f"Node '{node_id}': no data file columns found. "
                f"Provide 'input_file' or 'precip_file'+'pet_file'."
            )

        if inputs_df is None or inputs_df.empty:
            raise ValueError(f"Node '{node_id}': failed to load input data.")

        self._replace_sentinels(inputs_df)
        if observed_arr is not None:
            sentinel_mask = np.isin(observed_arr, self.missing_values)
            observed_arr = observed_arr.astype(float)
            observed_arr[sentinel_mask] = np.nan

        area = float(row.get('area_km2', 0))

        return CatchmentData(
            node_id=node_id,
            inputs=inputs_df,
            observed=observed_arr,
            dates=inputs_df.index,
            area_km2=area,
            metadata={'source_files': source_files},
        )

    # ------------------------------------------------------------------
    # Internal: helpers
    # ------------------------------------------------------------------
    def _resolve_path(self, rel: str) -> Path:
        p = Path(rel)
        if p.is_absolute():
            return p
        candidate = self.data_dir / p
        if candidate.exists():
            return candidate
        candidate2 = self.topology_csv.parent / p
        if candidate2.exists():
            return candidate2
        raise FileNotFoundError(
            f"Data file not found: '{rel}'. "
            f"Tried: {candidate}, {candidate2}"
        )

    def _read_csv_with_dates(self, path: Path) -> pd.DataFrame:
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]
        date_col = _find_column(df, DATE_SYNONYMS)
        if date_col is None:
            raise ValueError(
                f"No date column found in {path}. "
                f"Expected one of: {DATE_SYNONYMS}"
            )
        if self.date_format:
            df[date_col] = pd.to_datetime(df[date_col], format=self.date_format)
        else:
            df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, infer_datetime_format=True)
        df = df.set_index(date_col).sort_index()
        df.index.name = None
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.DatetimeIndex(df.index)
        return df

    def _extract_columns(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
        """From a single-file DataFrame, extract inputs and optional observed."""
        pcol = _find_column(df, PRECIP_SYNONYMS)
        ecol = _find_column(df, PET_SYNONYMS)
        fcol = _find_column(df, FLOW_SYNONYMS)

        if pcol is None:
            raise ValueError(f"No precipitation column found. Expected one of: {PRECIP_SYNONYMS}")
        if ecol is None:
            raise ValueError(f"No PET column found. Expected one of: {PET_SYNONYMS}")

        inputs = df[[pcol, ecol]].rename(columns={pcol: 'precipitation', ecol: 'pet'})
        observed = None
        if fcol:
            observed = df[fcol].values

        return inputs, observed

    def _replace_sentinels(self, df: pd.DataFrame) -> None:
        for col in df.columns:
            if df[col].dtype in (np.float64, np.float32, np.int64, np.int32, float, int):
                df[col] = df[col].replace(self.missing_values, np.nan)

    def _apply_missing_strategy(self) -> None:
        if self._data is None:
            return
        for cdata in self._data.values():
            if self.missing_strategy == 'interpolate':
                for col in cdata.inputs.columns:
                    cdata.inputs[col] = cdata.inputs[col].interpolate(
                        method='linear', limit=5
                    )
            elif self.missing_strategy == 'drop':
                cdata.inputs = cdata.inputs.dropna()
                if cdata.observed is not None:
                    valid_idx = cdata.inputs.index
                    obs_series = pd.Series(cdata.observed, index=cdata.dates)
                    obs_series = obs_series.reindex(valid_idx)
                    cdata.observed = obs_series.values
                    cdata.dates = valid_idx

    def _summarise_catchment(self, cdata: CatchmentData) -> CatchmentDataSummary:
        n = len(cdata.inputs)
        period = (
            str(cdata.dates[0].date()) if len(cdata.dates) > 0 else 'N/A',
            str(cdata.dates[-1].date()) if len(cdata.dates) > 0 else 'N/A',
        )
        p_miss = (cdata.inputs['precipitation'].isna().sum() / max(n, 1)) * 100 if 'precipitation' in cdata.inputs else 0
        e_miss = (cdata.inputs['pet'].isna().sum() / max(n, 1)) * 100 if 'pet' in cdata.inputs else 0

        q_miss = None
        if cdata.observed is not None:
            q_miss = (np.isnan(cdata.observed.astype(float)).sum() / max(len(cdata.observed), 1)) * 100

        longest = self._longest_gap(cdata.inputs)

        return CatchmentDataSummary(
            node_id=cdata.node_id,
            n_records=n,
            period=period,
            precip_missing_pct=p_miss,
            pet_missing_pct=e_miss,
            flow_missing_pct=q_miss,
            longest_gap_days=longest,
            source_files=cdata.metadata.get('source_files', []),
        )

    @staticmethod
    def _longest_gap(df: pd.DataFrame) -> int:
        """Longest contiguous NaN gap across all columns."""
        max_gap = 0
        for col in df.columns:
            mask = df[col].isna()
            if not mask.any():
                continue
            groups = (~mask).cumsum()
            gap_lengths = mask.groupby(groups).sum()
            if len(gap_lengths) > 0:
                max_gap = max(max_gap, int(gap_lengths.max()))
        return max_gap

    def _compute_junction_overlaps(self) -> Dict[str, JunctionOverlapInfo]:
        """Compute effective calibration periods at junction nodes."""
        if self._data is None or self._topo_df is None:
            return {}

        ds_map: Dict[str, Optional[str]] = {}
        for _, row in self._topo_df.iterrows():
            nid = str(row['id']).strip()
            ds = row.get('downstream_id', None)
            if pd.isna(ds) or ds == '':
                ds_map[nid] = None
            else:
                ds_map[nid] = str(ds).strip()

        parents: Dict[str, List[str]] = {nid: [] for nid in ds_map}
        for nid, ds in ds_map.items():
            if ds and ds in parents:
                parents[ds].append(nid)

        overlaps: Dict[str, JunctionOverlapInfo] = {}
        for jid, upstream_ids in parents.items():
            if not upstream_ids:
                continue

            jdata = self._data.get(jid)
            if jdata is None:
                continue

            j_obs_period = None
            j_start = jdata.dates[0] if len(jdata.dates) > 0 else None
            j_end = jdata.dates[-1] if len(jdata.dates) > 0 else None
            if jdata.observed is not None and j_start is not None:
                valid = ~np.isnan(jdata.observed.astype(float))
                if valid.any():
                    valid_dates = jdata.dates[valid]
                    j_start = valid_dates[0]
                    j_end = valid_dates[-1]
                    j_obs_period = (str(j_start.date()), str(j_end.date()))

            upstream_periods: Dict[str, Tuple[str, str]] = {}
            eff_start = j_start
            eff_end = j_end
            for uid in upstream_ids:
                udata = self._data.get(uid)
                if udata is None:
                    continue
                if len(udata.dates) == 0:
                    continue
                u_start = udata.dates[0]
                u_end = udata.dates[-1]
                upstream_periods[uid] = (str(u_start.date()), str(u_end.date()))

                if eff_start is not None:
                    eff_start = max(eff_start, u_start)
                if eff_end is not None:
                    eff_end = min(eff_end, u_end)

            eff_period = None
            eff_days = 0
            if eff_start is not None and eff_end is not None and eff_end > eff_start:
                eff_period = (str(eff_start.date()), str(eff_end.date()))
                eff_days = (eff_end - eff_start).days

            min_days = int(self.min_calibration_years * 365.25)
            is_suff = eff_days >= min_days

            overlaps[jid] = JunctionOverlapInfo(
                junction_id=jid,
                junction_observed_period=j_obs_period,
                upstream_periods=upstream_periods,
                effective_calibration_period=eff_period,
                effective_days=eff_days,
                is_sufficient=is_suff,
            )

        return overlaps


__all__ = [
    'CatchmentData',
    'CatchmentDataSummary',
    'JunctionOverlapInfo',
    'DataValidationReport',
    'NetworkDataLoader',
]
