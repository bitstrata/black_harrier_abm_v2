from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import rasterio
from rasterio.warp import transform
import pandas as pd


@dataclass
class RasterSamplers:
    habitat_raster: Optional[str] = None
    dem_raster: Optional[str] = None
    wind_u_raster: Optional[str] = None
    wind_v_raster: Optional[str] = None
    crs_epsg: int = 4326

    def _sample_raster(self, path: Optional[str], lat: float, lon: float) -> Optional[float]:
        if not path:
            return None
        with rasterio.open(path) as ds:
            # Transform WGS84 to raster CRS
            result = transform({"init": "epsg:4326"}, ds.crs, [lon], [lat])
            xs, ys = result[:2]
            row, col = ds.index(xs[0], ys[0])
            val = ds.read(1)[row, col]
            if np.isnan(val):
                return None
            return float(val)

    def habitat(self, lat: float, lon: float, t: pd.Timestamp) -> float:
        val = self._sample_raster(self.habitat_raster, lat, lon)
        if val is None:
            return 0.5
        # Assume habitat raster is already scaled 0..1; clamp for safety
        return max(0.0, min(1.0, float(val)))

    def slope(self, lat: float, lon: float) -> float:
        # Very simple slope proxy: finite diff on DEM; for production use a real gradient
        if not self.dem_raster:
            return 0.0
        with rasterio.open(self.dem_raster) as ds:
            result = transform({"init": "epsg:4326"}, ds.crs, [lon], [lat])
            xs, ys = result[:2]
            r, c = ds.index(xs[0], ys[0])
            elev = ds.read(1)
            r0, c0 = max(1, r), max(1, c)
            window = elev[r0 - 1 : r0 + 2, c0 - 1 : c0 + 2]
            gy, gx = np.gradient(window.astype(float))
            slope_mag = float(np.hypot(gx.mean(), gy.mean()))
            # Normalize to ~[0,1] then map to [-1,1] with sign ~ uphill penalty
            return -min(1.0, slope_mag / 50.0)

    def wind(self, lat: float, lon: float, t: pd.Timestamp) -> Tuple[float, float]:
        if not (self.wind_u_raster and self.wind_v_raster):
            return (0.0, 0.0)
        u = self._sample_raster(self.wind_u_raster, lat, lon) or 0.0
        v = self._sample_raster(self.wind_v_raster, lat, lon) or 0.0
        return (float(u), float(v))