"""Detector and sample geometry, consistent with LaueTools (top reflection, 'Z>0').

The detector is described by the 5 LaueTools calibration parameters
``[dd, xcen, ycen, xbet, xgam]`` (mm, pixel, pixel, deg, deg) plus pixel size
and frame shape.  Pixel <-> direction conversions reproduce
``LaueGeometry.calc_xycam`` / ``calc_uflab`` (checked in the tests), but are
written as an explicit ray-plane intersection so that the scattering point may
be anywhere (depth along the beam, finite footprint).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np

from .constants import KI_LT2

DEG = np.pi / 180.0


@dataclass
class DetectorGeometry:
    """Flat pixel detector in the LT2 frame (origin = calibration point I on the sample).

    Parameters
    ----------
    calib : [dd, xcen, ycen, xbet, xgam]
        LaueTools calibration (dd = distance I-O to the detector plane, mm).
    pixelsize : float
        mm (0.075 for EIGER2).
    shape : (n_rows, n_cols) = (n_y, n_x)
        EIGER2 X CdTe 4M: (2162, 2068), i.e. 162.2 x 155.1 mm2 (Y x X).
    """

    calib: Sequence[float]
    pixelsize: float = 0.075
    shape: Tuple[int, int] = (2162, 2068)
    label: str = "EIGER2 X CdTe 4M"

    def __post_init__(self):
        dd, xcen, ycen, xbet, xgam = [float(c) for c in self.calib]
        self.dd, self.xcen, self.ycen, self.xbet, self.xgam = dd, xcen, ycen, xbet, xgam
        cosb, sinb = np.sin(xbet * DEG), np.cos(xbet * DEG)      # LaueTools' cos/sin(beta)
        self.normal = np.array([0.0, cosb, sinb])                  # unit vector I->O
        self.IO = dd * self.normal
        u = np.array([1.0, 0.0, 0.0])
        w = np.array([0.0, sinb, -cosb])
        cg, sg = np.cos(-xgam * DEG), np.sin(-xgam * DEG)
        self.ex = cg * u + sg * w      # direction of increasing X pixel
        self.ey = -sg * u + cg * w     # direction of increasing Y pixel

    # -- constructors -----------------------------------------------------------
    @classmethod
    def from_lauetools_det(cls, path, shape=None, **kw):
        """Read a LaueTools ``.det`` file (first line: dd, xcen, ycen, xbet, xgam, pixelsize, dim1, dim2)."""
        with open(path) as f:
            vals = [float(v) for v in f.readline().replace(",", " ").split()]
        if shape is not None:
            shp = tuple(shape)
        elif len(vals) >= 8:
            shp = (int(vals[6]), int(vals[7]))
        else:
            shp = (2162, 2068)
        return cls(vals[:5], vals[5], shp, **kw)

    # -- pixel <-> lab ------------------------------------------------------------
    def pixel_to_lab(self, x, y):
        """Lab position (mm, LT2) of pixel coordinates (x, y) (continuous)."""
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        X1 = (x - self.xcen) * self.pixelsize
        Y1 = (y - self.ycen) * self.pixelsize
        return self.IO + X1[..., None] * self.ex + Y1[..., None] * self.ey

    def lab_to_pixel(self, M):
        OM = np.asarray(M, float) - self.IO
        return (self.xcen + OM @ self.ex / self.pixelsize,
                self.ycen + OM @ self.ey / self.pixelsize)

    def intersect(self, directions, sources=None):
        """Pixel coordinates where rays (sources + t * directions) hit the detector plane.

        Returns x, y (float pixels), path length (mm) and a mask of rays going
        towards the detector (t > 0).
        """
        d = np.atleast_2d(np.asarray(directions, float))
        d = d / np.linalg.norm(d, axis=-1, keepdims=True)
        S = np.zeros_like(d) if sources is None else np.broadcast_to(np.asarray(sources, float), d.shape)
        dn = d @ self.normal
        with np.errstate(divide="ignore", invalid="ignore"):
            t = ((self.IO - S) @ self.normal) / dn
        ok = (dn > 1e-9) & (t > 0)
        M = S + t[:, None] * d
        x, y = self.lab_to_pixel(M)
        return np.where(ok, x, np.nan), np.where(ok, y, np.nan), np.where(ok, t, np.nan), ok

    def directions_from_pixels(self, x, y, source=None):
        """Unit scattered vectors uf (LT2) and distances (mm) from a source point to pixels."""
        M = self.pixel_to_lab(x, y)
        IM = M - (0.0 if source is None else np.asarray(source, float))
        r = np.linalg.norm(IM, axis=-1)
        return IM / r[..., None], r

    def two_theta_chi(self, x, y, source=None):
        """2theta, chi (deg) with LaueTools conventions."""
        uf, _ = self.directions_from_pixels(x, y, source)
        tth = np.arccos(np.clip(uf[..., 1], -1, 1)) / DEG
        chi = np.arctan(-uf[..., 0] / (uf[..., 2] + 1e-17)) / DEG
        return tth, chi

    def in_frame(self, x, y, margin: float = 0.0):
        ny, nx = self.shape
        return (x >= -0.5 + margin) & (x < nx - 0.5 - margin) & (y >= -0.5 + margin) & (y < ny - 0.5 - margin)

    # -- radiometry ----------------------------------------------------------------
    def incidence_cos(self, directions):
        """cos(angle between ray and detector normal)."""
        d = np.asarray(directions, float)
        return np.abs(d @ self.normal) / np.linalg.norm(d, axis=-1)

    def pixel_solid_angle(self, x, y, source=None):
        """Solid angle (sr) of the pixel centred at (x, y) seen from the source."""
        uf, r = self.directions_from_pixels(x, y, source)
        return self.pixelsize ** 2 * np.abs(uf @ self.normal) / r ** 2

    def pixel_grid(self, step: int = 1):
        ny, nx = self.shape
        return np.meshgrid(np.arange(0, nx, step, dtype=float), np.arange(0, ny, step, dtype=float))


@dataclass
class SampleGeometry:
    """Flat sample surface through the calibration point.

    ``incidence_deg`` is the angle between the beam and the surface (40 deg on
    BM32).  The outward normal lies in the vertical plane containing the beam:
    n = (0, -sin a, cos a) in LT2.  ``thickness_um`` = inf for a thick crystal.
    """

    incidence_deg: float = 40.0
    thickness_um: float = np.inf

    @property
    def normal(self):
        a = self.incidence_deg * DEG
        return np.array([0.0, -np.sin(a), np.cos(a)])

    def gamma_in(self, ki=KI_LT2):
        """Direction cosine of the incident beam w.r.t. the inward normal (> 0)."""
        return -(np.asarray(ki) @ self.normal)

    def gamma_out(self, uf):
        """Direction cosine of the exit beam w.r.t. the outward normal (> 0 if it leaves the surface)."""
        return np.asarray(uf) @ self.normal

    def source_points(self, depth_um, ki=KI_LT2):
        """Scattering points (mm) for depths below the surface along the incident ray."""
        path_mm = np.asarray(depth_um, float) * 1e-3 / self.gamma_in(ki)
        return path_mm[..., None] * np.asarray(ki)
