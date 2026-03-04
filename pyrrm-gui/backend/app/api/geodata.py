"""
API routes for geospatial data management.

Handles shapefile uploads (catchment polygons, gauge points) and
serves them as GeoJSON for the web map.
"""

import json
import shutil
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, UploadFile, File, Form

from app.config import get_settings

router = APIRouter()

ALLOWED_LAYERS = {"catchments", "gauges"}


def _geodata_dir() -> Path:
    """Return the directory where geospatial layers are stored."""
    settings = get_settings()
    base = settings.data_dir / "geodata"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _geojson_path(layer: str) -> Path:
    return _geodata_dir() / f"{layer}.geojson"


def _convert_shapefile_to_geojson(shp_dir: Path, layer: str) -> Path:
    """
    Convert an uploaded shapefile (.shp + companions) to GeoJSON.

    Uses fiona/geopandas if available, falls back to a stub that stores
    the raw files until the dependencies are installed.
    """
    try:
        import geopandas as gpd
    except ImportError:
        raise HTTPException(
            status_code=501,
            detail=(
                "geopandas is not installed on the server. "
                "Install with: pip install geopandas fiona"
            ),
        )

    shp_files = list(shp_dir.glob("*.shp"))
    if not shp_files:
        raise HTTPException(status_code=400, detail="No .shp file found in upload")

    gdf = gpd.read_file(str(shp_files[0]))
    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    out = _geojson_path(layer)
    gdf.to_file(str(out), driver="GeoJSON")
    return out


@router.get("/layers")
async def list_layers():
    """List available geospatial layers with basic info."""
    layers = []
    for name in ALLOWED_LAYERS:
        path = _geojson_path(name)
        if path.exists():
            try:
                data = json.loads(path.read_text())
                n_features = len(data.get("features", []))
            except Exception:
                n_features = 0
            layers.append({
                "name": name,
                "available": True,
                "n_features": n_features,
            })
        else:
            layers.append({"name": name, "available": False, "n_features": 0})
    return {"layers": layers}


@router.get("/layers/{layer}")
async def get_layer_geojson(layer: str):
    """Return GeoJSON for a layer (catchments or gauges)."""
    if layer not in ALLOWED_LAYERS:
        raise HTTPException(status_code=400, detail=f"Unknown layer: {layer}")
    path = _geojson_path(layer)
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Layer '{layer}' has not been uploaded yet",
        )
    return json.loads(path.read_text())


@router.post("/layers/{layer}/upload")
async def upload_shapefile(
    layer: str,
    files: list[UploadFile] = File(...),
):
    """
    Upload a shapefile (multi-file: .shp, .dbf, .shx, .prj, etc.) or
    a single .geojson file. Converts to GeoJSON and stores on the server.
    """
    if layer not in ALLOWED_LAYERS:
        raise HTTPException(status_code=400, detail=f"Unknown layer: {layer}")

    if len(files) == 1 and files[0].filename and files[0].filename.endswith(".geojson"):
        content = await files[0].read()
        try:
            geojson = json.loads(content)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid GeoJSON")
        out = _geojson_path(layer)
        out.write_text(json.dumps(geojson))
        n_features = len(geojson.get("features", []))
        return {"layer": layer, "format": "geojson", "n_features": n_features}

    tmp_dir = _geodata_dir() / f"_upload_{layer}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir()

    try:
        extensions_found = set()
        for f in files:
            if not f.filename:
                continue
            ext = Path(f.filename).suffix.lower()
            extensions_found.add(ext)
            dest = tmp_dir / f.filename
            content = await f.read()
            dest.write_bytes(content)

        if ".shp" not in extensions_found:
            raise HTTPException(
                status_code=400,
                detail="Shapefile upload requires at least a .shp file (plus .dbf, .shx, .prj)",
            )

        out = _convert_shapefile_to_geojson(tmp_dir, layer)
        data = json.loads(out.read_text())
        n_features = len(data.get("features", []))
        return {"layer": layer, "format": "shapefile", "n_features": n_features}
    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)


@router.delete("/layers/{layer}")
async def delete_layer(layer: str):
    """Remove a geospatial layer."""
    if layer not in ALLOWED_LAYERS:
        raise HTTPException(status_code=400, detail=f"Unknown layer: {layer}")
    path = _geojson_path(layer)
    if path.exists():
        path.unlink()
    return {"detail": f"Layer '{layer}' deleted"}
