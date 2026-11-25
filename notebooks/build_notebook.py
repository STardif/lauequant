"""Builds notebooks/01_Ge_quantitative_laue.ipynb (run, then execute with nbconvert)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []


def md(s):
    C.append(nbf.v4.new_markdown_cell(s.strip()))


def code(s):
    C.append(nbf.v4.new_code_cell(s.strip()))


md(r"""
# Quantitative Laue microdiffraction on BM32 — first forward model (Ge single crystal)

Chain implemented in `lauequant` (see `PLAN.md`):

1. **Incident spectrum** — absolute flux per eV at the sample from the pyXRT beamline model, + polarisation, + KB convergence.
2. **Geometry** — LaueTools conventions (LT / LT2 frames, 5-parameter detector calibration), sample surface at 40°.
3. **Scattering** — energy-integrated reflectivity of each reflection *and its harmonics*: kinematic (mosaic) and dynamical (perfect crystal, Bragg case, with absorption).
4. **Detector** — EIGER2 X CdTe 4M: interaction probability, charge sharing at a low threshold, Cd/Te K-fluorescence escape, parallax in the sensor, count-rate saturation.
5. **Background** — Ge K fluorescence.

Everything is in photons/s or counts/s for the full white beam (200 mA, S3 = 0.2×0.2 mm²) unless stated.
""")

code(r"""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath(".."))                     # lauequant
os.environ.setdefault("PYXRT_PATH", os.path.abspath("../../ext/pyXRT"))   # <- adapt to your pyXRT clone
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30); pd.set_option("display.precision", 4)
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False,
                     "axes.prop_cycle": plt.cycler(color=["#257de0", "#f47230", "#00b894", "#f2a600",
                                                          "#ee7fac", "#008a17", "#6a4ce2"])})
import lauequant as lq
from lauequant import (IncidentSpectrum, FocusedBeam, DetectorGeometry, SampleGeometry, Crystal,
                       PhotonCountingDetector, LaueSimulator, fluorescence_background)
""")

md("## 1. Incident spectrum (pyXRT)")
code(r"""
try:
    spec = IncidentSpectrum.from_pyxrt(vg1=0.22, hg1=0.15, vg3=0.2, hg3=0.2, current_a=0.2,
                                       kbv_angle_mrad=3.0, kbh_angle_mrad=3.0)
except ImportError:
    print("pyXRT not found: using the cached spectrum (same settings)")
    spec = IncidentSpectrum.from_csv("../data/bm32_spectrum_200mA_S3_0.2x0.2.csv")
E = spec.energies_kev
print(f"total flux      : {spec.total_flux():.3e} photons/s   (LAUEMAX draft: 9.5e11)")
print(f"mean energy     : {spec.mean_energy():.2f} keV          (draft: 18.6)")
print(f"band (1e-3 max) : {spec.band(1e-3)}")
print(f"sigma fraction  : min {spec.sigma_fraction.min():.5f}  (linear polarisation, horizontal)")
beam = FocusedBeam.bm32(vg3_mm=0.2, hg3_mm=0.2)
print(f"KB convergence  : {beam.convergence_v_mrad:.2f} mrad (V) x {beam.convergence_h_mrad:.2f} mrad (H)")

fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
ax[0].plot(E, spec.flux_per_ev * 1e3 / 1e10)
ax[0].set(xlabel="photon energy (keV)", ylabel="flux (1e10 photons/s/keV)", title="Flux per unit energy (enters all integrals)")
ax[1].semilogy(E, spec.flux_per_ev * 1e3)
ax[1].set(xlabel="photon energy (keV)", ylabel="photons/s/keV", ylim=(1e6, 1e11), title="Same, log scale")
plt.tight_layout()
""")

md(r"""
## 2. Geometry

Detector: EIGER2 X CdTe 4M (2068 × 2162 pixels of 75 µm). The calibration below is a *typical* LaueMAX top-reflection
geometry (replace with your `.det` file: `DetectorGeometry.from_lauetools_det(path, shape=(2162, 2068))`).
The orientation matrix is the one refined by LaueTools on the Ge reference crystal of `LaueImages/Ge_blanc_0000_LT_5.fit`.
Pixel ↔ direction conversions are checked against `LaueTools.LaueGeometry.calc_xycam / calc_uflab` in `tests/`.
""")
code(r"""
ge = Crystal.germanium()
UB = np.array([[-0.071476709, -0.99582593, -0.056760883],
               [ 0.720618491, -0.012210383, -0.693224276],
               [ 0.689637637, -0.090452332,  0.718483337]])
geo = DetectorGeometry([79.5, 1030.2, 1110.7, 0.35, -0.42], pixelsize=0.075, shape=(2162, 2068))
sample = SampleGeometry(incidence_deg=40.0)
det = PhotonCountingDetector(threshold_kev=4.0)
sim = LaueSimulator(ge, UB, spec, geo, sample=sample, beam=beam, detector=det, model="dynamical")
sim_kin = LaueSimulator(ge, UB, spec, geo, sample=sample, beam=beam, detector=det, model="kinematic")
print(f"Ge: V = {ge.volume:.2f} A^3, density = {ge.density:.3f} g/cm3, energy window {sim.emin:.2f}-{sim.emax:.2f} keV")
try:   # cross-check spot positions with LaueTools
    from LaueTools import lauecore as LC
    r = sim.reflections()
    hkl = r[["h", "k", "l"]].values[:50]
    out = LC.calcSpots_fromHKLlist(UB, ge.B0, hkl, dict(CCDparam=geo.calib, pixelsize=0.075, dim=(2162, 2068)))
    print("max |dX|, |dY| vs LaueTools (pixel):", np.abs(out[6] - r.X.values[:50]).max(), np.abs(out[7] - r.Y.values[:50]).max())
except ImportError:
    pass
""")

md(r"""
## 3. Scattering model: kinematic vs dynamical

For a stationary crystal the diffracted rate of reflection *h* is $P_h = \Phi_E(E_h)\,E_h\,\zeta_h$ with
$\zeta_h=\int R\,d\lambda/\lambda$.  A Ge wafer is a *perfect* crystal: the dynamical $\zeta$ (thick crystal,
Bragg case, absorption, asymmetry $b=-\gamma_0/\gamma_h$, σ/π polarisation) is the relevant one; the kinematic value
(mosaic crystal, absorption-limited volume) overestimates strong, low-order reflections, the more so below the
Ge K edge (11.10 keV) where absorption is weak.
""")
code(r"""
r = sim.reflections()
rk = sim_kin.reflections().set_index(["h", "k", "l"])
r = r.assign(kin_over_dyn=r.zeta_kin / r.zeta_dyn)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
below = r.E_keV < 11.103
ax[0].scatter(r.E_keV[~below], r.kin_over_dyn[~below], s=10, label="above Ge K edge")
ax[0].scatter(r.E_keV[below], r.kin_over_dyn[below], s=10, label="below Ge K edge")
ax[0].set(yscale="log", xlabel="reflection energy (keV)", ylabel="zeta_kin / zeta_dyn", title="Extinction: kinematic overestimate")
ax[0].legend()
ax[1].scatter(r.ext_depth_um, r.abs_depth_um, s=10)
ax[1].plot([0.1, 100], [0.1, 100], color="#666666", lw=1)
ax[1].set(xscale="log", yscale="log", xlabel="extinction depth (um)", ylabel="absorption depth (um)",
          title="Points above the line: extinction-dominated")
plt.tight_layout()
print(f"sum over spots: kinematic / dynamical = {sim_kin.spots().photons_s.sum() / sim.spots().photons_s.sum():.2f}")
""")

md("## 4. Spot list (photons/s per spot, harmonics included)")
code(r"""
spots = sim.spots()
print(f"{len(spots)} spots on the detector, {len(sim.reflections())} reflections; "
      f"{spots.photons_s_at_det.sum():.3e} photons/s diffracted onto the detector "
      f"({spots.photons_s_at_det.sum() / spec.total_flux():.1e} of the incident flux)")
spots.head(15)[["h0", "k0", "l0", "X", "Y", "two_theta", "chi", "photons_s", "harmonics", "h_main", "k_main", "l_main",
                "E_main_keV", "E_mean_keV"]]
""")
code(r"""
fig, ax = plt.subplots(figsize=(6.6, 6.2))
sc = ax.scatter(spots.X, spots.Y, s=4 + 60 * (np.log10(spots.photons_s) - 3).clip(0) ** 1.5,
                c=spots.E_mean_keV, cmap="cividis", edgecolor="none")
ax.set(xlim=(0, 2068), ylim=(2162, 0), aspect="equal", xlabel="X (pixel)", ylabel="Y (pixel)",
       title="Predicted Laue pattern — size: log photons/s, colour: mean energy")
plt.colorbar(sc, label="photon-weighted mean energy (keV)", shrink=0.8)
for _, s in spots.head(8).iterrows():
    ax.annotate(f"{s.h0:.0f}{s.k0:.0f}{s.l0:.0f}", (s.X, s.Y), xytext=(6, 6), textcoords="offset points", fontsize=8)
""")

md(r"""
## 5. Detector response (EIGER2 X CdTe, 750 µm)

Per photon: interaction depth (and lateral parallax at oblique incidence), Cd/Te K fluorescence (reabsorbed in a
neighbour or escaping), charge cloud (diffusion during drift) shared between pixels, then each pixel counts if its
charge exceeds the threshold.  With the threshold at **4 keV** a 20 keV photon absorbed within ~10 µm of a pixel
border is counted twice: the mean number of counts per photon exceeds 1 and depends on energy.

*`charge_sigma0_um`, `bias_v`, `noise_kev` and `dead_time_ns` are first guesses to be calibrated (see PLAN.md).*
""")
code(r"""
Eg = np.arange(5, 40.1, 0.5)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
ax[0].plot(Eg, det.interaction_probability(Eg), label="interaction probability (normal incidence)")
ax[0].plot(Eg, det.interaction_probability(Eg, np.cos(np.radians(45))), label="at 45 deg incidence")
ax[0].set(xlabel="photon energy (keV)", ylim=(0.9, 1.005), title="CdTe 750 um efficiency"); ax[0].legend()
for thr in (4.0, 8.0, 12.0):
    d = PhotonCountingDetector(threshold_kev=thr)
    ax[1].plot(Eg, d.flat_response(Eg, n=15000), label=f"threshold {thr:.0f} keV")
ax[1].axhline(1, color="#666666", lw=1)
ax[1].axvline(26.71, color="#999999", lw=0.8, ls=":"); ax[1].axvline(31.81, color="#999999", lw=0.8, ls=":")
ax[1].set(xlabel="photon energy (keV)", ylabel="counts per incident photon", title="Counting multiplicity (flat illumination)")
ax[1].legend(); plt.tight_layout()
""")
code(r"""
top = spots.head(4)
fig, ax = plt.subplots(2, 4, figsize=(13, 6.2))
rows = []
for j, (_, s) in enumerate(top.iterrows()):
    key = (s.h0, s.k0, s.l0)
    for i, thr in enumerate((4.0, 12.0)):
        sim.detector = PhotonCountingDetector(threshold_kev=thr)
        res = sim.spot_counts(key, roi_half=4, rng=1)
        im = ax[i, j].imshow(res["roi"], norm=LogNorm(vmin=1e3, vmax=1e7), cmap="cividis",
                             extent=(res["x0"] - .5, res["x0"] + 8.5, res["y0"] + 8.5, res["y0"] - .5))
        ax[i, j].set_title(f"{key[0]:.0f}{key[1]:.0f}{key[2]:.0f}  thr {thr:.0f} keV", fontsize=9)
        ax[i, j].grid(False)
        rows.append(dict(spot=f"{key[0]:.0f}{key[1]:.0f}{key[2]:.0f}", threshold_keV=thr, photons_s=res["photons_s"],
                         counts_s=res["counts_s"], counts_per_photon=res["counts_s"] / res["photons_s"],
                         peak_pixel_cps=res["roi"].max(),
                         peak_measured_cps=sim.detector.measured_rate(res["roi"].max())))
sim.detector = det
fig.colorbar(im, ax=ax, label="counts/s per pixel", shrink=0.8)
pd.DataFrame(rows)
""")
md(r"""
The brightest pixels of the strongest spots receive a few 10⁶ counts/s at full flux: the retrigger dead time already
loses 10–20 % there, and the vendor count-rate correction (tabulated for monochromatic beams) is only approximately
valid.  Attenuating the beam or closing S3 for intensity work on a perfect crystal is therefore advisable.
""")

md("## 6. Background: Ge K fluorescence")
code(r"""
bg, lines, grid = fluorescence_background(sim, step=32, return_lines=True)
for k, (el, img) in lines.items():
    if img.max() > 0:
        print(f"{k:8s} {el:7.3f} keV   median {np.median(img):9.1f} counts/s/pixel")
print(f"total: median {np.median(bg):.0f}, max {bg.max():.0f} counts/s/pixel")
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(bg, cmap="cividis"); ax.grid(False)
ax.set(title="Ge K fluorescence (counts/s/pixel, threshold 4 keV)", xlabel="X", ylabel="Y")
plt.colorbar(im, shrink=0.8)
""")
md(r"""
With the threshold at 4 keV, the Ge Kα/Kβ fluorescence (9.9/11.0 keV, excited by all photons above 11.1 keV)
gives ~10⁴ counts/s on every pixel — a uniform background that a measured Ge image must show.  Raising the threshold
above 11 keV removes it (and the soft part of the Laue spectrum with it).  This is a simple, sensitive check of the
absolute flux × detector efficiency, independent of the diffraction model.
""")

md("## 7. Simulated detector image (0.1 s exposure, Poisson noise)")
code(r"""
out = sim.image(exposure_s=0.1, background=bg, noise=True, rng=3)
print("pixels above the count-rate-correction cutoff:", int(out["saturated"].sum()))
from scipy.ndimage import maximum_filter
img = out["counts"]
fig, ax = plt.subplots(1, 2, figsize=(12, 5.6))
ax[0].imshow(maximum_filter(img, size=9) + 1, norm=LogNorm(vmin=1e3, vmax=3e5), cmap="cividis")
ax[0].set(title="Full frame (9x9 max-filtered for display)", xlabel="X (pixel)", ylabel="Y (pixel)")
o = geo.pixel_origin                      # array index = calibration coordinate - 1
x0, y0 = int(round(spots.X[0] - o)) - 60, int(round(spots.Y[0] - o)) - 60
ax[1].imshow(img[y0:y0 + 120, x0:x0 + 120] + 1, norm=LogNorm(vmin=300, vmax=3e5), cmap="cividis",
             extent=(x0 - .5, x0 + 119.5, y0 + 119.5, y0 - .5))
ax[1].set(title="Zoom on the strongest spot (raw pixels)", xlabel="X (pixel)")
for a in ax: a.grid(False)
plt.tight_layout()
""")

md(r"""
## 8. Smoke test on real data (old MarCCD Ge pattern from the LaueTools repository)

`Ge_blanc_0000_LT_5.fit` contains 95 indexed Ge spots with integrated intensities measured with the former MarCCD
(phosphor + CCD: signal ∝ absorbed energy) on the pre-EBS source.  Neither the spectrum nor the detector are those
of LaueMAX, so only the *ranking* and the energy trend of the residuals are meaningful here; the real test needs an
EIGER2 image of Ge with known S3 openings, ring current and exposure.
""")
code(r"""
from scipy.stats import spearmanr
from scipy.spatial import cKDTree
from lauequant.materials import Medium
cols = ("spot_index Itot h k l Xtheo Ytheo Xexp Yexp XdevCalib YdevCalib pixDevCalib 2theta_theo chi_theo Energy "
        "PeakAmplitude Imax PeakBkg PeakFwhm1 PeakFwhm2 PeakTilt XdevPeakFit YdevPeakFit").split()
expt = pd.DataFrame(np.loadtxt("../data/Ge_blanc_0000_LT_5.fit", comments="#"), columns=cols)
geo_ccd = DetectorGeometry([70.775, 941.736, 1082.562, 0.63, -0.682], 0.079142, (2048, 2048), label="MarCCD165")
gos = Medium.from_formula("Gd2O2S", 7.3)          # phosphor, ~40 um (assumed)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for model in ("dynamical", "kinematic"):
    s2 = LaueSimulator(ge, UB, spec, geo_ccd, sample=sample, beam=beam, model=model).reflections()
    s2 = s2.assign(sig=s2.photons_s_at_det * s2.E_keV * -np.expm1(-gos.mu(s2.E_keV.values) * 40e-4))
    g = s2.groupby(["h0", "k0", "l0"]).agg(X=("X", "first"), Y=("Y", "first"), sig=("sig", "sum"),
                                         E=("E_keV", lambda e: e.iloc[0])).reset_index()
    d, i = cKDTree(g[["X", "Y"]].values).query(expt[["Xtheo", "Ytheo"]].values)
    I, S, Ew = expt.Itot.values, g.sig.values[i], expt.Energy.values
    k = np.exp(np.median(np.log(I / S)))
    print(f"{model:10s}: {np.sum(d < 0.5)}/{len(d)} spots matched (max {d.max():.3f} px), "
          f"Spearman = {spearmanr(I, S)[0]:.3f}, rms log10 residual = {np.std(np.log10(I / S / k)):.2f}")
    ax[0].loglog(S * k, I, "o", ms=3, label=model)
    ax[1].semilogy(Ew, I / (S * k), "o", ms=3, label=model)
ax[0].plot([1e3, 1e5], [1e3, 1e5], color="#666666", lw=1)
ax[0].set(xlabel="predicted (scaled)", ylabel="measured Itot (MarCCD)", title="Ge_blanc: measured vs predicted"); ax[0].legend()
ax[1].axhline(1, color="#666666", lw=1)
ax[1].set(xlabel="energy of the fundamental (keV)", ylabel="measured / predicted", title="Residual vs energy (spectrum shape check)")
ax[1].legend(); plt.tight_layout()
""")

md(r"""
The ranking is reproduced (Spearman ≈ 0.8, all spots matched to 0.01 px), but the residuals show a strong,
smooth energy trend: the old set-up delivered relatively fewer photons above ~17 keV than the LaueMAX spectrum used
here (different mirrors and cut-off), and the phosphor efficiency is only guessed.  This is exactly the kind of
smooth T(E) correction that stage 6 of the plan fits on an EIGER2 Ge image.
""")

md(r"""
## 9. Next: comparison with an EIGER2 Ge image

Sketch of the calibration loop (to be implemented in `lauequant.fit`, see PLAN.md, stage 6):

1. read the image + `vg1 hg1 vg3 hg3`, ring current, exposure from the BLISS file (`pyXRT.bm32_from_h5` already does it for the spectrum);
2. predict ROIs (`sim.spot_counts`) and integrate measured ROIs at the same pixels, subtracting the local background;
3. fit a global scale *k* (flux × efficiency) and a smooth spectral correction *T(E)* (e.g. KB angle, mirror roughness) on the ratio measured/predicted, using spots whose harmonics are well separated in energy;
4. use the Ge fluorescence level (threshold scan 4 → 12 keV) as an independent check of *k*;
5. determine the charge-sharing parameter from the ratio of spot integrals at 4 keV and ~E/2 thresholds.
""")

nb["cells"] = C
nb["metadata"]["kernelspec"] = dict(name="python3", display_name="Python 3", language="python")
nbf.write(nb, "01_Ge_quantitative_laue.ipynb")
print("written")
