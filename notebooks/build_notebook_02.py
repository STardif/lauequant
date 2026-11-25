"""Builds notebooks/02_Ge_EIGER_comparison.ipynb."""
import nbformat as nbf

C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# Ge reference on LaueMAX (EIGER2 X CdTe, 12 Nov 2026): measurement vs absolute model

Data: `RefGe_GeRefH2_12Nov_17h48_RefGe.h5`, scan 1.1 (0.1 s, single image `scan0001/eiger4m_0000.h5`),
calibration and UB from `calib.det` (LaueTools, Ge, 98 indexed spots).  The image, slits (S1, S3), mirror angles and
ring current are read with pyXRT (`bliss_hdf5_reader.BlissScan`, `xrf_beamline.bm32_from_h5`).

Pixel coordinates: X, Y in the `.dat`, `.fit` and `.det` files follow the LaueTools "XMAS" convention (first pixel
= (1, 1), a legacy of the Fit2D/XMAS era), so the array element is `img[Y - 1, X - 1]`.  `DetectorGeometry.pixel_origin = 1`
handles this; all lauequant positions are in the calibration frame, ROIs give `col0, row0` for array slicing.

Measured spot intensities: own ROI integration (11 × 11 pixels around the *predicted* position, local background =
median of a square annulus at 8–12 px), the same box being used for the prediction.  (LaueTools' `Itot` in the `.dat`
is the single-pixel maximum for spots where the 2D fit did not converge, so it is not used.)
""")
code(r"""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath(".."))
os.environ.setdefault("PYXRT_PATH", os.path.abspath("../../ext/pyXRT"))        # <- your pyXRT clone
DATA = os.environ.get("GE_DATA", os.path.abspath("../../gedata"))           # <- folder with the master file and scan0001/
MASTER = os.path.join(DATA, "master.h5")            # RefGe_GeRefH2_12Nov_17h48_RefGe.h5
DETFILE = os.path.join(DATA, "calib.det")
SCAN = "1.1"
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.ndimage import maximum_filter
from scipy.stats import spearmanr
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30); pd.set_option("display.precision", 3)
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False,
                     "axes.prop_cycle": plt.cycler(color=["#257de0", "#f47230", "#00b894", "#f2a600", "#6a4ce2"])})
from lauequant import fluorescence_background, PhotonCountingDetector
from lauequant.compare import read_scan, build_simulator, compare_spots, fit_scale, block_median
img, info = read_scan(MASTER, SCAN)
mask = img > 0                                       # module gaps are 0
T = info["exposure_s"]
info
""")

md("## 1. Model with the beamline settings of the scan (pyXRT defaults, KB at 3.0 mrad) — no free parameter")
code(r"""
sim30 = build_simulator(MASTER, SCAN, DETFILE, kb_angle_mrad=3.0)
print("S1/S3 (mm):", {k: round(v, 3) for k, v in sim30.spectrum.meta["settings"].items() if k[:2] in ("vg", "hg")},
      f"current {info['current_ma']:.1f} mA, threshold {info['threshold_kev']:.1f} keV")
print(f"flux at sample {sim30.spectrum.total_flux():.3e} ph/s, mean energy {sim30.spectrum.mean_energy():.2f} keV, "
      f"KB convergence {sim30.beam.convergence_v_mrad:.2f} x {sim30.beam.convergence_h_mrad:.2f} mrad")
sp30 = compare_spots(sim30, img, T)
k30, rms30, n30 = fit_scale(sp30)
print(f"{n30} well-measured spots: measured / predicted = {k30:.3f} (median), rms {rms30:.3f} dex")

fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(maximum_filter(img, 5) + 1, norm=LogNorm(vmin=100, vmax=1e5), cmap="cividis"); ax.grid(False)
on = sp30.pred_counts > 3000
o = sim30.detector_geometry.pixel_origin        # .dat/.det coordinates: first pixel = (1, 1)
ax.scatter(sp30.X[on] - o, sp30.Y[on] - o, s=60, facecolors="none", edgecolors="#f47230", lw=0.8, label="predicted, > 3000 counts")
ax.set(title=f"Scan {SCAN} ({T} s), 5x5 max-filtered", xlabel="X (pixel)", ylabel="Y (pixel)"); ax.legend(loc="lower right")
""")
code(r"""
sel = (sp30.meas_counts.notna()) & (sp30.pred_counts > 3000)
fig, ax = plt.subplots(figsize=(8, 3.8))
ax.semilogy(sp30.E_mean_keV[sel], (sp30.meas_counts / sp30.pred_counts)[sel].clip(1e-3), "o", ms=4)
ax.axhline(k30, color="#666666", lw=1)
ax.set(xlabel="photon-weighted mean energy of the spot (keV)", ylabel="measured / predicted",
       title="KB 3.0 mrad: flat ratio up to ~24 keV, then the real spectrum is cut off earlier", ylim=(1e-2, 3))
""")
md(r"""
Without any adjustable parameter the model predicts the spots within a factor 0.65 and the measured/predicted ratio is
flat from 9 to ~24 keV — the shape of the spectrum, the dynamical reflectivities and the detector efficiency are
consistent over that range.  Above ~24 keV the measured intensities collapse: the high-energy cut-off of the real
beam is lower than computed with the KB mirrors at 3.0 mrad (and much lower than for the nominal 2.8 mrad).

## 2. Fitting the cut-off: effective KB grazing angle
""")
code(r"""
rows, sims = [], {}
for ang in (3.0, 3.2, 3.3, 3.35, 3.4, 3.45, 3.5, 3.6):
    s = build_simulator(MASTER, SCAN, DETFILE, kb_angle_mrad=ang)
    c = compare_spots(s, img, T)
    k, rms, n = fit_scale(c)
    rows.append(dict(kb_mrad=ang, scale_k=k, rms_dex=rms, n=n, flux=s.spectrum.total_flux(),
                     flux_times_k=k * s.spectrum.total_flux(), mean_E=s.spectrum.mean_energy()))
    sims[ang] = (s, c)
scan_tab = pd.DataFrame(rows); scan_tab
""")
code(r"""
best = scan_tab.loc[scan_tab.rms_dex.idxmin(), "kb_mrad"]
sim, sp = sims[best]
k, rms, n = fit_scale(sp)
simk = build_simulator(MASTER, SCAN, DETFILE, kb_angle_mrad=best, model="kinematic")
spk = compare_spots(simk, img, T)
kk, rmsk, nk = fit_scale(spk)
print(f"best effective KB angle {best} mrad: scale k = {k:.3f}, rms = {rms:.3f} dex ({100*(10**rms-1):.0f} %), n = {n}")
print(f"same spectrum, kinematic model: k = {kk:.3f}, rms = {rmsk:.3f} dex")
sel = (sp.meas_counts > 2000) & (sp.pred_counts > 3000)
selk = (spk.meas_counts > 2000) & (spk.pred_counts > 3000)
print("Spearman (dynamical):", round(spearmanr(sp.meas_counts[sel], sp.pred_counts[sel])[0], 4))
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
ax[0].loglog(kk * spk.pred_counts[selk], spk.meas_counts[selk], "o", ms=4, label=f"kinematic (rms {rmsk:.2f} dex)")
ax[0].loglog(k * sp.pred_counts[sel], sp.meas_counts[sel], "o", ms=4, label=f"dynamical (rms {rms:.2f} dex)")
ax[0].plot([1e3, 1e6], [1e3, 1e6], color="#666666", lw=1)
ax[0].set(xlabel=f"predicted counts in {T} s (x k)", ylabel="measured counts", title=f"KB {best} mrad: spot integrals"); ax[0].legend()
ax[1].semilogy(sp.E_mean_keV[sel], (sp.meas_counts / (k * sp.pred_counts))[sel], "o", ms=4, label="dynamical")
ax[1].axhline(1, color="#666666", lw=1)
ax[1].set(xlabel="photon-weighted mean energy (keV)", ylabel="measured / (k x predicted)", ylim=(0.3, 3),
          title="Residuals vs energy"); ax[1].legend()
plt.tight_layout()
""")
code(r"""
s = sp[sel].copy(); s["ratio"] = s.meas_counts / (k * s.pred_counts)
g = sim.reflections().groupby(["h0", "k0", "l0"]).gamma_out.first()
s = s.join(g, on=["h0", "k0", "l0"])
def binned(col, bins):
    t = s.groupby(pd.cut(s[col], bins), observed=True).ratio
    return pd.DataFrame(dict(n=t.size(), median_ratio=t.median()))
display(binned("E_mean_keV", [9, 11.1, 13, 15, 17, 19, 21, 23, 25, 27]))
display(binned("two_theta", [50, 65, 80, 95, 110, 125, 140]))
display(binned("pred_counts", [3e3, 1e4, 3e4, 1e5, 3e5, 1e6]))
""")
md(r"""
No residual trend with 2θ, exit angle or intensity; the remaining scatter (~20 % rms) includes ROI integration on
weak spots and the simple shape of the cut-off (residuals +20 % at 23–25 keV, −15 % at 25–27 keV).

## 3. Incident spectrum implied by the data
""")
code(r"""
E = sim.spectrum.energies_kev
fig, ax = plt.subplots(figsize=(8, 3.8))
ax.plot(E, sims[3.0][0].spectrum.flux_per_ev * 1e3, label="pyXRT, KB 3.0 mrad (scan settings)")
ax.plot(E, k * sim.spectrum.flux_per_ev * 1e3, label=f"fitted: KB {best} mrad x k = {k:.2f}")
ax.set(xlabel="photon energy (keV)", ylabel="photons/s/keV", title="Incident spectrum at the sample, scan 1.1", xlim=(4, 36))
ax.legend()
fitted = sim.spectrum.scaled(k)
lo, hi = fitted.band(0.5)
print(f"fitted spectrum: {fitted.total_flux():.3e} ph/s at {info['current_ma']:.0f} mA "
      f"(pyXRT KB 3.0: {sims[3.0][0].spectrum.total_flux():.3e}), mean {fitted.mean_energy():.2f} keV, "
      f"half-maximum band (per eV) {lo:.1f}-{hi:.1f} keV")
""")

md(r"""
## 4. Diffuse background: Ge K fluorescence
With the threshold at 9.0 keV the Ge Kα line (9.89 keV) is counted only when ≥ 91 % of its charge stays in one pixel,
so the predicted background depends strongly on the charge-cloud size (`charge_sigma0_um`); the Kβ line (10.98 keV)
much less.  Below: block medians (32 × 32 px) of the image vs the predicted fluorescence with the fitted spectrum.
""")
code(r"""
mb = block_median(img, mask, 32)
B = 32
rows = []
for sig0 in (4, 8, 12):
    s2 = build_simulator(MASTER, SCAN, DETFILE, kb_angle_mrad=best, charge_sigma0_um=sig0)
    s2.spectrum = s2.spectrum.scaled(k)
    bg = fluorescence_background(s2, step=32) * T
    pb = block_median(bg, np.ones_like(bg, bool), 32)
    ok = np.isfinite(mb)
    r = mb[ok] / pb[ok]
    rows.append(dict(charge_sigma0_um=sig0, flat_resp_Ka=s2.detector.flat_response([9.886])[0],
                     pred_median=np.median(pb[ok]), meas_median=np.median(mb[ok]), ratio_median=np.median(r),
                     ratio_p16=np.percentile(r, 16), ratio_p84=np.percentile(r, 84)))
    if sig0 == 8:
        pb8 = pb
pd.DataFrame(rows)
""")
code(r"""
fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
im = ax[0].imshow(mb / pb8, vmin=0.5, vmax=1.5, cmap="RdBu_r"); ax[0].grid(False)
ax[0].set(title="measured / predicted background (sigma0 = 8 um)", xlabel="X block", ylabel="Y block"); plt.colorbar(im, ax=ax[0])
ok = np.isfinite(mb)
ax[1].plot(pb8[ok], mb[ok], ".", ms=3)
ax[1].plot([0, pb8[ok].max()], [0, pb8[ok].max()], color="#666666", lw=1)
ax[1].set(xlabel=f"predicted Ge K fluorescence (counts / {T} s / pixel)", ylabel="measured block median", title="Background")
plt.tight_layout()
""")
md(r"""
With the default charge-cloud size the fluorescence background comes out at ~1.0 × the measurement once the spectrum
scale fitted on the spots is applied, i.e. spots and background give the same flux × efficiency.  Because of the
threshold sitting just below Ge Kα, this is not yet an independent check: a threshold scan (or a threshold of 4–6 keV)
would make it one.

## 5. Spot shape
""")
code(r"""
def moments(roi):
    roi = np.clip(roi, 0, None); n = roi.shape[0]; yy, xx = np.mgrid[0:n, 0:n]; s = roi.sum()
    mx, my = (roi * xx).sum() / s, (roi * yy).sum() / s
    return np.sqrt((roi * (xx - mx) ** 2).sum() / s), np.sqrt((roi * (yy - my) ** 2).sum() / s), roi.max() / s
strong = sp[sp.meas_counts > 5e4].sort_values("meas_counts", ascending=False).head(25)
rows, ex = [], None
for r in strong.itertuples():
    p = sim.spot_counts((r.h0, r.k0, r.l0), roi_half=4, rng=0, n_v=11, n_h=31)
    m = img[p["row0"]:p["row0"] + 9, p["col0"]:p["col0"] + 9] - r.meas_bkg
    rows.append(moments(m) + moments(p["roi"]))
    if ex is None:
        ex = (r, m, p["roi"] * T)
mom = np.median(np.array(rows), 0)
print("rms width X, Y (pixel) and peak/integral — measured:", mom[:3].round(3), " predicted:", mom[3:].round(3))
print("extra broadening (quadrature): X %.2f px, Y %.2f px" % tuple(np.sqrt(np.clip(mom[:2] ** 2 - mom[3:5] ** 2, 0, None))))
r, m, pr = ex
fig, ax = plt.subplots(1, 2, figsize=(8, 3.6))
for a, d, t in ((ax[0], m, "measured"), (ax[1], pr, "predicted (x 1/k not applied)")):
    im = a.imshow(np.clip(d, 1, None), norm=LogNorm(vmin=30, vmax=max(m.max(), pr.max())), cmap="cividis"); a.grid(False)
    a.set_title(f"{r.h0}{r.k0}{r.l0}: {t}", fontsize=9)
plt.colorbar(im, ax=ax, shrink=0.85)
""")
md(r"""
The measured spots are broader than the model (extra ~0.5–0.7 px rms in both directions, i.e. ~0.4–0.5 mrad seen from
the sample).  Integrated intensities are not affected, but the peak pixel rates (count-rate effects) are.  Candidates:
real KB convergence/aberrations, residual curvature/mosaicity of the reference crystal, detector point-spread at
9 keV threshold.  A spot-shape fit vs S3 opening would separate them.

""")
md(r"""
## 6. Depth (penetration) effects on spot shape and position

Each harmonic is traced from a depth drawn from the dynamical depth profile (`depth_model="darwin"`): inside the
Darwin plateau the wavefield is extinguished within ~Λ₀/2π, in the tails (~¼ of the integrated reflectivity) only
absorption limits it.  A scattering point at depth z is displaced by z/γ₀ along the beam, i.e. by
≈ (z/γ₀)·sin2θ perpendicular to the exit ray: on the detector the spot moves and stretches **radially (along 2θ)**.
""")
code(r"""
def radial_unit(row, geo):
    tth, chi = np.radians(row.two_theta), np.radians(row.chi)
    uf = lambda a: np.array([-np.sin(a) * np.sin(chi), np.cos(a), np.sin(a) * np.cos(chi)])
    x1, y1, _, _ = geo.intersect(uf(tth - 1e-4)); x2, y2, _, _ = geo.intersect(uf(tth + 1e-4))
    u = np.array([x2[0] - x1[0], y2[0] - y1[0]]); return u / np.linalg.norm(u)
def widths_rt(roi, u):
    roi = np.clip(roi, 0, None); n = roi.shape[0]; yy, xx = np.mgrid[0:n, 0:n]; s = roi.sum()
    dx, dy = xx - (roi * xx).sum() / s, yy - (roi * yy).sum() / s
    rad, tan = dx * u[0] + dy * u[1], -dx * u[1] + dy * u[0]
    return np.sqrt((roi * rad ** 2).sum() / s), np.sqrt((roi * tan ** 2).sum() / s)
sim_nd = build_simulator(MASTER, SCAN, DETFILE, kb_angle_mrad=best); sim_nd.depth_model = "none"
refl = sim.reflections(); rows = []
for r in sp[sp.meas_counts > 3e4].itertuples():
    key = (r.h0, r.k0, r.l0); u = radial_unit(r, sim.detector_geometry)
    main = refl[(refl.h0 == r.h0) & (refl.k0 == r.k0) & (refl.l0 == r.l0)].sort_values("photons_s").iloc[-1]
    d = dict(hkl=f"{r.h0}{r.k0}{r.l0}", E_main=main.E_keV, ext_um=main.ext_depth_um, abs_um=main.abs_depth_um,
             mean_depth_um=sim.mean_depth_um(main))
    for lbl, s_ in (("none", sim_nd), ("depth", sim)):
        p = s_.spot_counts(key, roi_half=4, rng=0, n_depth=8)
        w = p["rays"]["w"]
        d[f"rad_{lbl}"], d[f"tan_{lbl}"] = widths_rt(p["roi"], u)
        d[f"cen_{lbl}"] = np.average(p["rays"]["x"], weights=w) * u[0] + np.average(p["rays"]["y"], weights=w) * u[1]
    m = img[p["row0"]:p["row0"] + 9, p["col0"]:p["col0"] + 9] - r.meas_bkg
    d["rad_meas"], d["tan_meas"] = widths_rt(m, u)
    rows.append(d)
dw = pd.DataFrame(rows); dw["radial_shift_px"] = dw.cen_depth - dw.cen_none
print(f"{len(dw)} strong spots; mean scattering depth {dw.mean_depth_um.median():.1f} um (range {dw.mean_depth_um.min():.1f}-{dw.mean_depth_um.max():.1f})")
print(f"radial centroid shift from depth: mean {dw.radial_shift_px.mean():.2f} px, spread (std) {dw.radial_shift_px.std():.2f} px")
med = dw[["rad_none", "rad_depth", "rad_meas", "tan_none", "tan_depth", "tan_meas"]].median()
print("median rms widths (px):"); print(med.round(3).to_string())
xr = np.sqrt(max(med.rad_meas ** 2 - med.rad_depth ** 2, 0)); xt = np.sqrt(max(med.tan_meas ** 2 - med.tan_depth ** 2, 0))
print(f"unexplained broadening (quadrature): radial {xr:.2f} px, tangential {xt:.2f} px")
fig, ax = plt.subplots(figsize=(6, 3.8))
ax.plot(dw.mean_depth_um, dw.rad_depth - dw.rad_none, "o", ms=4)
ax.set(xlabel="mean scattering depth (um)", ylabel="radial width added by depth (px)", title="Depth contribution to spot width")
dw.sort_values("mean_depth_um").round(3).tail(8)
""")
md(r"""
Mean depths of 2–15 µm (the tails of the Darwin curve reach the absorption depth) widen the spots radially by only a few
hundredths of a pixel (up to ~0.2 px for the deepest), and shift them by ~0.05–0.3 px along 2θ (towards lower 2θ).  The extra measured
broadening is *larger tangentially than radially*, so it is not a depth effect (a weak positive correlation of the
radial excess with depth, Spearman ≈ 0.3, suggests depth plays a minor part).

The **position** effect matters more than the width: the mean shift is absorbed by the calibration (dd, centre), but
its spot-to-spot spread (~0.07 px rms, up to 0.2 px between shallow and deep reflections, i.e. up to ~1.5e-4 rad at 100 mm) is of the order of the deviatoric-strain resolution and
differs between reflections and harmonics.  It is below the present LaueTools residuals on this image (0.3 px radial,
0.4 px tangential, uncorrelated with the predicted shift), but would show with better peak fitting.

## 7. Summary
* **Absolute agreement without free parameter:** measured / predicted = 0.65 with pyXRT's spectrum at KB 3.0 mrad.
* **High-energy cut-off:** the spot intensities fix it.  An effective KB grazing angle of ≈ 3.4 mrad (instead of 3.0
  in pyXRT, 2.8 nominal) gives a flat residual; scale k ≈ 0.75 (flux × efficiency), i.e. ≈ 7.2e11 ph/s at 193 mA
  in this configuration.
* **Dynamical vs kinematic:** the perfect-crystal model halves the scatter (0.08 vs 0.16 dex): the Ge reference is
  extinction-limited, as expected.
* **Background:** consistent with the same scale, but depends on charge sharing at a 9 keV threshold.
* **Spot shape:** measured spots ~1.3–1.5 × wider than modelled, more tangentially than radially: not a depth effect.
* **Depth:** mean scattering depths 2–15 µm → spots shifted by ~0.05–0.3 px along 2θ (spread ~0.07 px between strong spots), radial widths +0.03 px (≤ 0.2 px): included, but minor.
""")

nb = nbf.v4.new_notebook(); nb["cells"] = C
nb["metadata"]["kernelspec"] = dict(name="python3", display_name="Python 3", language="python")
nbf.write(nb, "02_Ge_EIGER_comparison.ipynb"); print("written")
