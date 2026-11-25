"""Consistency tests.  Run with:  PYXRT_PATH=/path/to/pyXRT pytest -q tests"""
import os

import numpy as np
import pytest

from lauequant import (Crystal, DetectorGeometry, IncidentSpectrum, LaueSimulator,
                       PhotonCountingDetector, M_LT_TO_LT2)
from lauequant.constants import HC_KEV_A
from lauequant.scattering import (darwin_zeta_nonabsorbing, dynamical_theta_integral,
                                  dynamical_zeta, kinematic_zeta)

try:
    from LaueTools import LaueGeometry as LTGeo
    from LaueTools import CrystalParameters as CP
    from LaueTools import lauecore as LC
    HAVE_LT = True
except Exception:  # pragma: no cover
    HAVE_LT = False

CALIB = [79.5, 1030.2, 1110.7, 0.35, -0.42]
UB_GE = np.array([[-0.071476709, -0.99582593, -0.056760883],
                  [0.720618491, -0.012210383, -0.693224276],
                  [0.689637637, -0.090452332, 0.718483337]])


@pytest.mark.skipif(not HAVE_LT, reason="LaueTools not importable")
def test_pixel_direction_roundtrip_vs_lauetools():
    geo = DetectorGeometry(CALIB, 0.075, (2162, 2068))
    rng = np.random.default_rng(1)
    x, y = rng.uniform(0, 2068, 200), rng.uniform(0, 2162, 200)
    tth_lt, chi_lt = LTGeo.calc_uflab(x, y, CALIB, pixelsize=0.075)
    tth, chi = geo.two_theta_chi(x, y)
    assert np.allclose(tth, tth_lt, atol=1e-9) and np.allclose(chi, chi_lt, atol=1e-9)
    uf = LTGeo.uflab_from2thetachi(tth_lt, chi_lt)
    xl, yl, _ = LTGeo.calc_xycam(uf, CALIB, pixelsize=0.075)
    xm, ym, _, ok = geo.intersect(uf)
    assert ok.all() and np.allclose(xm, xl, atol=1e-7) and np.allclose(ym, yl, atol=1e-7)


@pytest.mark.skipif(not HAVE_LT, reason="LaueTools not importable")
def test_source_offset_vs_lauetools():
    geo = DetectorGeometry(CALIB, 0.075, (2162, 2068))
    uf = LTGeo.uflab_from2thetachi(np.array([60.0, 90.0, 120.0]), np.array([-20.0, 5.0, 30.0]))
    xl, yl, _ = LTGeo.calc_xycam(uf, CALIB, offset=0.05, pixelsize=0.075)
    xm, ym, _, _ = geo.intersect(uf, sources=np.array([0, 0.05, 0]))
    assert np.allclose(xm, xl, atol=1e-7) and np.allclose(ym, yl, atol=1e-7)


@pytest.mark.skipif(not HAVE_LT, reason="LaueTools not importable")
def test_B0_vs_lauetools():
    ge = Crystal.germanium()
    assert np.allclose(ge.B0, CP.calc_B_RR(ge.lattice), atol=1e-12)
    tri = Crystal("tri", [4.1, 5.2, 6.3, 80, 95, 110])
    assert np.allclose(tri.B0, CP.calc_B_RR(tri.lattice), atol=1e-10)


@pytest.mark.skipif(not HAVE_LT, reason="LaueTools not importable")
def test_spot_positions_vs_lauetools():
    ge = Crystal.germanium()
    hkl = np.array([[3, 3, 3], [2, 2, 4], [4, 2, 2], [5, 3, 3], [1, 3, 5]])
    H, K, L, Qx, Qy, Qz, X, Y, tth, chi, E = LC.calcSpots_fromHKLlist(
        UB_GE, ge.B0, hkl, dict(CCDparam=CALIB, pixelsize=0.075, dim=(2162, 2068), kf_direction="Z>0"))
    geo = DetectorGeometry(CALIB, 0.075, (2162, 2068))
    q = hkl @ (M_LT_TO_LT2 @ UB_GE @ ge.B0).T
    Em = HC_KEV_A * np.sum(q ** 2, 1) / (-2 * q[:, 1])
    uf = np.array([0, 1, 0]) + q * (HC_KEV_A / Em)[:, None]
    xm, ym, _, _ = geo.intersect(uf)
    assert np.allclose(xm, X, atol=1e-6) and np.allclose(ym, Y, atol=1e-6)
    assert np.allclose(Em, E * HC_KEV_A / 12.398, rtol=1e-9)


def test_darwin_nonabsorbing_limit():
    ge = Crystal.germanium()
    for hkl, E, g0, gh in [((1, 1, 1), 15.0, None, None), ((8, 8, 0), 20.0, 0.3, 0.9), ((9, 7, 7), 25.0, 0.9, 0.2)]:
        th = np.arcsin(HC_KEV_A / (2 * ge.d_spacing([hkl])[0] * E))
        g0 = g0 or np.sin(th)
        gh = gh or np.sin(th)
        F = abs(ge.structure_factor([hkl], E)[0])
        ch = ge.chi(F, E) + 0j
        c0 = ge.chi(ge.F0(E)[0].real, E) + 0j
        for C in (1.0, np.cos(2 * th)):
            z = dynamical_theta_integral(c0, ch, ch, th, g0, gh, C) / np.tan(th)
            assert z == pytest.approx(darwin_zeta_nonabsorbing(F, E, th, ge.volume, g0, gh, C), rel=1e-3)


def test_dynamical_tends_to_kinematic_for_weak_reflections():
    """Very weak (high-order, hot) reflection: extinction << absorption -> dyn ~ kin."""
    ge = Crystal.germanium(B_iso=3.0)
    hkl, E = (16, 12, 8), 26.0
    th = np.arcsin(HC_KEV_A / (2 * ge.d_spacing([hkl])[0] * E))
    g0 = gh = np.sin(th)
    F = ge.structure_factor([hkl], E)[0]
    Fb = ge.structure_factor([[-16, -12, -8]], E)[0]
    zd = dynamical_zeta(ge.chi(ge.F0(E)[0], E), ge.chi(F, E), ge.chi(Fb, E), th, g0, gh, 1, 0)
    zk = kinematic_zeta(F, E, th, ge.volume, ge.mu(E)[0], g0, gh, 1, 0)
    assert zd == pytest.approx(zk, rel=0.05)


def test_detector_mono_half_threshold_counts_about_once():
    det = PhotonCountingDetector(threshold_kev=10.0)
    m = det.multiplicity([20.0], n=40000)[0]
    assert 0.85 < m < 1.05
    det_low = PhotonCountingDetector(threshold_kev=4.0)
    assert det_low.multiplicity([20.0], n=40000)[0] > 1.1     # double counting at low threshold


@pytest.mark.skipif("PYXRT_PATH" not in os.environ, reason="set PYXRT_PATH to the pyXRT folder")
def test_flux_matches_lauemax_paper():
    sp = IncidentSpectrum.from_pyxrt(vg1=0.22, hg1=0.15, vg3=0.2, hg3=0.2, current_a=0.2)
    assert 8.5e11 < sp.total_flux() < 1.1e12          # paper: 9.5e11 photons/s at 200 mA
    assert 17.5 < sp.mean_energy() < 19.5             # paper: 18.6 keV
    assert sp.sigma_fraction.min() > 0.99


def test_simulator_runs_and_photon_budget():
    e = np.arange(4, 40, 0.05)
    flux = 6e7 * np.exp(-0.5 * ((e - 18) / 5) ** 2)     # ~1e12 photons/s, toy spectrum
    sp = IncidentSpectrum(e, flux)
    sim = LaueSimulator(Crystal.germanium(), UB_GE, sp, DetectorGeometry(CALIB))
    s = sim.spots()
    assert len(s) > 20
    top = s.iloc[0]
    res = sim.spot_counts((top.h0, top.k0, top.l0), rng=0)
    # counts / photons = interaction probability x multiplicity: between 0.9 and 2
    assert 0.9 < res["counts_s"] / res["photons_s"] < 2.0


def test_read_lauetools_det(tmp_path):
    from lauequant.compare import read_lauetools_det
    p = tmp_path / "c.det"
    p.write_text("100.134, 1086.96, 1179.4, 0.395, 0.399, 0.075, 2162, 2068\nheader\nOrientation Matrix:\n"
                 "[[0.25,-0.78,-0.56],[0.17,0.61,-0.77],[0.95,0.09,0.28]]\n")
    calib, pix, shape, UB = read_lauetools_det(p)
    assert calib == [100.134, 1086.96, 1179.4, 0.395, 0.399] and pix == 0.075 and shape == (2162, 2068)
    assert UB.shape == (3, 3)
