"""
API routes for Batch Analysis.

Provides endpoints for loading batch calibration results from disk,
computing diagnostics, and generating comparison plots.
"""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

from app.config import get_settings
from app.schemas.analysis import (
    LoadSessionRequest,
    SessionSummary,
    SessionDetailResponse,
    GaugeSummary,
    ExperimentInfo,
    DiagnosticsResponse,
    SummaryTableRow,
)
from app.services.analysis import analysis_service

router = APIRouter()


# ── Signature reference (static data) ────────────────────────────────────────

@router.get("/signatures/reference")
async def get_signature_reference():
    """
    Get complete reference documentation for all hydrological signatures.
    
    Returns metadata for all 47 signatures organized by category, including:
    - name: Human-readable name
    - category: Signature category (Magnitude, Variability, etc.)
    - units: Measurement units
    - range: Valid value range
    - description: Brief description
    - formula: Mathematical formula or algorithm
    - interpretation: How to interpret values
    - related: Related signature IDs
    - references: Literature references
    """
    return analysis_service.get_signature_reference()


# ── Available batches (server-side directory scan) ───────────────────────────

@router.get("/batches")
async def list_available_batches():
    """
    List batch result folders available on the server.

    Scans the configured batch_results directory for subfolders.  Each
    subfolder that contains at least one ``batch_result.pkl`` (recursively)
    is returned as an available batch the user can load.
    """
    settings = get_settings()
    root = settings.get_batch_results_dir()
    if not root.is_dir():
        return {"root": str(root), "batches": []}

    batches = []
    try:
        for entry in sorted(root.iterdir()):
            if not entry.is_dir() or entry.name.startswith("."):
                continue
            pkl_count = len(list(entry.rglob("batch_result.pkl")))
            batches.append({
                "name": entry.name,
                "path": str(entry),
                "pkl_count": pkl_count,
            })
    except OSError:
        pass

    return {"root": str(root), "batches": batches}


# ── Sessions ─────────────────────────────────────────────────────────────────

@router.post("/sessions", response_model=SessionSummary)
async def load_session(request: LoadSessionRequest):
    """Load batch results from a folder into a new analysis session."""
    try:
        session = analysis_service.load_session(
            folder_path=request.folder_path,
            name=request.name,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return SessionSummary(
        id=session.id,
        name=session.name,
        folder_path=session.folder_path,
        gauge_ids=session.gauge_ids,
        total_experiments=session.total_experiments,
        total_failures=session.total_failures,
        loaded_at=session.loaded_at,
    )


@router.get("/sessions", response_model=List[SessionSummary])
async def list_sessions():
    """List all loaded analysis sessions."""
    return [
        SessionSummary(
            id=s.id,
            name=s.name,
            folder_path=s.folder_path,
            gauge_ids=s.gauge_ids,
            total_experiments=s.total_experiments,
            total_failures=s.total_failures,
            loaded_at=s.loaded_at,
        )
        for s in analysis_service.list_sessions()
    ]


@router.get("/sessions/{session_id}", response_model=SessionDetailResponse)
async def get_session(session_id: str):
    """Get detailed session information."""
    try:
        session = analysis_service.get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")

    gauges = []
    for gid in session.gauge_ids:
        gs = analysis_service.get_gauge_summary(session_id, gid)
        gauges.append(GaugeSummary(**gs))

    return SessionDetailResponse(
        id=session.id,
        name=session.name,
        folder_path=session.folder_path,
        gauge_ids=session.gauge_ids,
        total_experiments=session.total_experiments,
        total_failures=session.total_failures,
        loaded_at=session.loaded_at,
        gauges=gauges,
    )


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Unload a session from memory."""
    try:
        analysis_service.delete_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"detail": "Session deleted"}


# ── Summary table ────────────────────────────────────────────────────────────

@router.get("/sessions/{session_id}/summary", response_model=List[SummaryTableRow])
async def get_summary(session_id: str):
    """Combined summary table across all gauges."""
    try:
        rows = analysis_service.get_combined_summary(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")
    return [SummaryTableRow(**r) for r in rows]


# ── Gauges ───────────────────────────────────────────────────────────────────

@router.get("/sessions/{session_id}/gauges", response_model=List[GaugeSummary])
async def list_gauges(session_id: str):
    """List gauges in a session with summary info."""
    try:
        session = analysis_service.get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")

    return [
        GaugeSummary(**analysis_service.get_gauge_summary(session_id, gid))
        for gid in session.gauge_ids
    ]


# ── Gauge experiments ────────────────────────────────────────────────────────

@router.get("/sessions/{session_id}/gauges/{gauge_id}/experiments", response_model=List[ExperimentInfo])
async def get_gauge_experiments(session_id: str, gauge_id: str):
    """List experiments for a gauge with headline metrics."""
    try:
        exps = analysis_service.get_experiments(session_id, gauge_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return [ExperimentInfo(**e) for e in exps]


# ── Diagnostics ──────────────────────────────────────────────────────────────

@router.get("/sessions/{session_id}/gauges/{gauge_id}/diagnostics")
async def get_diagnostics(session_id: str, gauge_id: str):
    """Full diagnostics: raw table, normalised table, clustermap, top experiments."""
    try:
        return analysis_service.get_diagnostics(session_id, gauge_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ── Comparison plots ─────────────────────────────────────────────────────────

@router.get("/sessions/{session_id}/gauges/{gauge_id}/comparison/hydrograph")
async def comparison_hydrograph(
    session_id: str,
    gauge_id: str,
    log_scale: bool = Query(False),
    experiments: Optional[str] = Query(None, description="Comma-separated experiment keys"),
):
    """Multi-experiment hydrograph comparison (Plotly JSON)."""
    exp_keys = experiments.split(",") if experiments else None
    try:
        return analysis_service.build_comparison_hydrograph(
            session_id, gauge_id, log_scale=log_scale, experiment_keys=exp_keys
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/sessions/{session_id}/gauges/{gauge_id}/comparison/fdc")
async def comparison_fdc(
    session_id: str,
    gauge_id: str,
    log_scale: bool = Query(True),
    experiments: Optional[str] = Query(None, description="Comma-separated experiment keys"),
):
    """Multi-experiment FDC comparison (Plotly JSON)."""
    exp_keys = experiments.split(",") if experiments else None
    try:
        return analysis_service.build_comparison_fdc(
            session_id, gauge_id, log_scale=log_scale, experiment_keys=exp_keys
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/sessions/{session_id}/gauges/{gauge_id}/comparison/scatter")
async def comparison_scatter(
    session_id: str,
    gauge_id: str,
    log_scale: bool = Query(False),
    experiments: Optional[str] = Query(None, description="Comma-separated experiment keys"),
):
    """Multi-experiment scatter comparison (Plotly JSON)."""
    exp_keys = experiments.split(",") if experiments else None
    try:
        return analysis_service.build_comparison_scatter(
            session_id, gauge_id, log_scale=log_scale, experiment_keys=exp_keys
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/sessions/{session_id}/gauges/{gauge_id}/comparison/signatures")
async def comparison_signatures(
    session_id: str,
    gauge_id: str,
    experiments: Optional[str] = Query(None, description="Comma-separated experiment keys"),
):
    """
    Hydrological signature comparison (Plotly JSON).

    Returns signature values and percent errors organized by category
    (Magnitude, Variability, Timing, FDC, Frequency, Recession, Baseflow,
    Event, Seasonality), along with pre-built radar charts, bar charts,
    and an overall error heatmap.
    """
    exp_keys = experiments.split(",") if experiments else None
    try:
        return analysis_service.build_signature_comparison(
            session_id, gauge_id, experiment_keys=exp_keys
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ── Individual report card ───────────────────────────────────────────────────

@router.get("/sessions/{session_id}/gauges/{gauge_id}/experiments/{exp_key}/report-card")
async def experiment_report_card(session_id: str, gauge_id: str, exp_key: str):
    """Full Plotly report card for a single experiment."""
    try:
        return analysis_service.get_report_card(session_id, gauge_id, exp_key)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ── Report card export ────────────────────────────────────────────────────────

@router.get("/export/sections")
async def get_export_sections():
    """Get available export sections with descriptions."""
    from pyrrm.visualization.report_export import get_export_sections
    return get_export_sections()


@router.get("/sessions/{session_id}/gauges/{gauge_id}/experiments/{exp_key}/report-card/export")
async def export_single_report_card(
    session_id: str,
    gauge_id: str,
    exp_key: str,
    format: str = Query("pdf", pattern="^(pdf|html|interactive)$"),
    sections: Optional[str] = Query(None, description="Comma-separated section IDs to include"),
):
    """
    Export a single experiment's report card.
    
    Args:
        session_id: Session identifier
        gauge_id: Gauge identifier
        exp_key: Experiment key
        format: Export format:
            - 'pdf': Landscape A4 single-page PDF
            - 'html': Landscape static HTML (for printing)
            - 'interactive': Interactive HTML with Plotly figures
        sections: Comma-separated section IDs (default: all sections)
    
    Returns:
        File download (PDF or HTML)
    """
    try:
        content, filename, media_type = analysis_service.export_report_card(
            session_id, gauge_id, exp_key,
            format=format,
            sections=sections.split(",") if sections else None
        )
        return Response(
            content=content,
            media_type=media_type,
            headers={"Content-Disposition": f'attachment; filename="{filename}"'}
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ImportError as e:
        raise HTTPException(status_code=501, detail=str(e))


@router.get("/sessions/{session_id}/gauges/{gauge_id}/report-card/export")
async def export_batch_report_card(
    session_id: str,
    gauge_id: str,
    format: str = Query("pdf", pattern="^(pdf|html|interactive)$"),
    experiments: Optional[str] = Query(None, description="Comma-separated experiment keys"),
    sections: Optional[str] = Query(None, description="Comma-separated section IDs to include"),
):
    """
    Export multiple experiments' report cards as a single document.
    
    Args:
        session_id: Session identifier
        gauge_id: Gauge identifier
        format: Export format:
            - 'pdf': Landscape A4, one page per experiment
            - 'html': Landscape static HTML with page breaks
            - 'interactive': Interactive HTML with Plotly figures
        experiments: Comma-separated experiment keys (default: all experiments)
        sections: Comma-separated section IDs (default: all sections)
    
    Returns:
        File download (PDF or HTML)
    """
    try:
        exp_keys = experiments.split(",") if experiments else None
        section_list = sections.split(",") if sections else None
        
        content, filename, media_type = analysis_service.export_batch_report_card(
            session_id, gauge_id,
            experiment_keys=exp_keys,
            format=format,
            sections=section_list
        )
        return Response(
            content=content,
            media_type=media_type,
            headers={"Content-Disposition": f'attachment; filename="{filename}"'}
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ImportError as e:
        raise HTTPException(status_code=501, detail=str(e))
