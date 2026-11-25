# Quantitative Laue microdiffraction on BM32/LaueMAX — development plan

Goal: predict, in **absolute counts per pixel**, the Laue pattern recorded by the EIGER2 X CdTe 4M on LaueMAX, starting from a Ge single crystal, so that spot intensities (and later shapes) can be used quantitatively (structure factors, composition, defect content, spectral calibration).

Forward model = product of independent, separately testable blocks:

```
Phi_E(E) [photons/s/eV]          pyXRT beamline model (+ polarisation, KB convergence)
  x geometry                       LaueTools conventions (LT/LT2 frames, .det calibration, sample at 40 deg)
  x zeta_h(E)                      kinematic or dynamical integrated reflectivity, all harmonics
  x air, sensor efficiency         xraylib
  x pixel response                 charge sharing, K-escape, parallax, threshold(s), dead time
  + backgrounds                    sample fluorescence, (Compton, TDS, air scatter)
  = expected counts per pixel  ->  compared with measured images (fit of a few calibration parameters)
```

## Status of the prototype (`lauequant` v0.1, Nov 2026)

| Block | Module | State |
|---|---|---|
| Incident spectrum, absolute | `spectrum.IncidentSpectrum.from_pyxrt` | wraps pyXRT `BM32_WHITE_BEAM` / `bm32_from_h5`; 9.8e11 ph/s, mean 18.4 keV at paper settings (draft: 9.5e11, 18.6) |
| Polarisation | `spectrum.bending_magnet_sigma_fraction` | sigma/pi split of the accepted fan; > 0.9997 linear on BM32 |
| KB convergence | `spectrum.FocusedBeam.bm32` | 0.56 (V) x 1.67 (H) mrad for S3 = 0.2 x 0.2 mm; uniform; traced per ray |
| Detector geometry | `geometry.DetectorGeometry` | ray-plane intersection, identical to `calc_xycam`/`calc_uflab` to 1e-12 px (tests) |
| Sample geometry | `geometry.SampleGeometry` | flat surface at 40 deg, thick or finite slab, depth -> parallax |
| Crystal | `crystal.Crystal` | B0 = LaueTools `calc_B_RR` (tests), F(hkl,E) with f0, f', f'' (xraylib), Debye-Waller; Ge, Si factories |
| Scattering | `scattering` | kinematic (absorbing slab) and dynamical (thick perfect crystal, Bragg case, absorption, asymmetry, sigma/pi), Darwin limit and dyn->kin limit tested |
| Harmonics | `simulate.LaueSimulator.spots` | all n(hkl) in the band summed per spot |
| Pixel response | `detector.PhotonCountingDetector` | Monte Carlo per photon: depth/parallax, Cd/Te K fluorescence + escape, charge cloud sharing, 2 thresholds, noise; non-paralysable dead time; vendor-correction cutoff |
| Background | `background.fluorescence_background` | Ge K fluorescence (thick sample, self-absorption, solid angle, air, detector response) |
| Validation on data | notebook §8 | old MarCCD Ge pattern: 95/95 spots at < 0.013 px, Spearman 0.8, strong smooth energy trend in residuals |

First physical conclusions from the prototype (Ge, 200 mA, S3 0.2 x 0.2 mm, dd 80 mm):

* A perfect Ge crystal is extinction-limited: kinematic/dynamical ~ 2 on the whole pattern, up to ~8 for strong reflections just below the Ge K edge (e.g. 422 at 11.0 keV). **The dynamical model must be the default for Ge/Si references.**
* Strongest spots: ~1e7 photons/s over ~6 pixels, ~3-4e6 counts/s in the brightest pixel: already 15-20 % dead-time loss. Intensity work on Ge should use reduced S3 openings or an attenuator.
* With a 4 keV threshold, **charge sharing makes the counts per photon energy dependent (~0.5 at 5 keV, ~1.2 at 12 keV, ~1.5 at 25 keV)**; spots appear ~20-30 % too strong and the error depends on their spectral content. This is the main "photon-counting detector used in white beam" effect, and is the first detector parameter to calibrate.
* Cd K escape produces a ~10 % dip in counts per photon just above 26.7 keV.
* Ge Kα/Kβ fluorescence gives ~1e4 counts/s/pixel at 4 keV threshold: a uniform, model-independent check of flux x efficiency.

## First comparison with LaueMAX data (Ge ref., 12 Nov 2026, scan 1.1, 0.1 s, EIGER2 threshold 9 keV)

Notebook `02_Ge_EIGER_comparison.ipynb`, module `lauequant.compare` (image and settings read with pyXRT).

* **No free parameter** (pyXRT spectrum with the recorded S1/S3/M1/M2/current, KB 3.0 mrad): measured/predicted = 0.65,
  flat from 9 to ~24 keV, collapsing above ~24 keV.
* **Fitted:** effective KB grazing angle 3.4 mrad (sets the high-energy cut-off) and a global scale k = 0.75:
  122 spots, rms 0.08 dex (~20 %), Spearman 0.99, no trend with 2theta, exit angle or intensity.
  Implied flux at the sample 7.2e11 ph/s at 193 mA; the product k x flux is stable (7.2-7.8e11) whatever the KB angle.
  Half-maximum band (per eV) 9.9-24.3 keV, mean 17.3 keV (draft paper: 12.5-27.5 keV, 18.6 keV).
* **Dynamical model confirmed:** kinematic rms 0.16 dex vs dynamical 0.08 dex.
* **Background:** Ge K fluorescence predicted at 0.98 x measured with the same scale and charge_sigma0 = 8 um; but at a
  9 keV threshold the Ka line is near threshold, so this fixes charge sharing rather than flux. Spatial residual:
  centre 0.8, corners 1.2-1.3, plus module-to-module steps (threshold dispersion) -> obliquity in the sensor and/or an
  extra broad background (Compton, scattering) to model.
* **Spot shape:** measured spots wider than modelled by ~0.7 (X) and ~0.6 (Y) px rms in quadrature (~0.45 mrad).
* **Depth:** rays start at depths drawn from the dynamical depth profile resolved over the Darwin curve
  (`depth_model="darwin"`: extinction-limited on the plateau, absorption-limited in the tails). Mean depths 2-15 um
  -> spots shifted by ~0.05-0.3 px along 2theta, radial widths +0.03 px (<= 0.2 px). The unexplained broadening is
  larger tangentially (0.78 px) than radially (0.61 px): not a depth effect. The spot-to-spot spread of the
  depth shift (~0.07 px rms) is near the strain resolution -> to include in strain refinement (LaueTools).
* **Pixel convention:** LaueTools .dat/.fit/.det coordinates use the XMAS convention (first pixel = (1, 1));
  `DetectorGeometry.pixel_origin = 1` maps them to numpy indices (array = coordinate - 1). Fixing a 1-px slip in the
  ROI extraction changed the results only marginally (best effective KB angle 3.40-3.45 mrad, k = 0.75-0.77,
  rms 0.083 dex). Remaining offset: image centroids are 0.37 px towards -X from LaueTools Gaussian-fit positions
  (Y: 0.02 px) -> centroid vs peak of spots elongated along X, to keep in mind when comparing shapes.
* Not yet tested: dead time (brightest pixel 1.1e6 cps), threshold dependence, other exposures/positions (scans 2.1-6.1).

## Development stages

### Stage 1 — Incident spectrum (done: wrapper; to do: calibration hooks)
* Done: absolute Phi_E(E) from pyXRT, ring current, slit gaps, KB angles; polarisation; filters/attenuators.
* To do:
  - read slits/current/exposure per scan from BLISS h5 (`IncidentSpectrum.from_pyxrt(beamline=bm32_from_h5(...))`) and normalise by the monitor;
  - correlated KB effect: the KB grazing angle varies along the mirror, so the high-energy cut-off varies across the convergent fan (spot shape vs energy); propagate per ray instead of averaging;
  - parametrised spectral correction T(E; theta_KB, roughness, a smooth spline) to be fitted (stage 6);
  - independent spectral checks: rainbow-filter dips (absolute E), foil K-yields (pyXRT), calibrated diode.

### Stage 2 — Geometry (done; to do: refinements)
* Done: LaueTools-compatible detector/sample frames; `.det` reader; depth parallax; convergence.
* To do: read `.fit` files directly (UB, calib, measured intensities); detector distortion map (pyFAI spline, as in the draft §2.6); transmission geometry and thin samples (Laue case); sample on a furnace/Be dome (extra absorbers on incident and exit paths).

### Stage 3 — Scattering (Ge done; to do: generalisation)
* Done: kinematic & dynamical (thick, Bragg case), anomalous dispersion across the Ge K edge, DW, harmonics, polarisation.
* To do:
  - finite-thickness perfect crystals (Darwin-Prins slab, Bragg and Laue cases) for films/membranes;
  - real (imperfect) crystals: mosaic + primary/secondary extinction (Zachariasen / Becker-Coppens), so that one parameter interpolates kinematic <-> dynamical;
  - strain gradients / curvature (spot streaking) via per-voxel UB (link with DAXM and LaueTools refinement output);
  - multigrain and multiphase (sum of grains, volumes, LaueNN indexation results);
  - CIF input (gemmi/pymatgen) for arbitrary structures; occupancy and composition fitting.

### Stage 4 — Detector response (prototype done; calibration needed)
* Done: per-photon Monte Carlo with charge sharing, K-escape, parallax, thresholds, dead time.
* To do:
  - **calibrate** `charge_sigma0_um`, `bias_v`, `noise_kev` from (i) threshold scans on a Ge pattern (spot integrals vs threshold, 4 -> E/2), (ii) knife-edge / spot profiles, (iii) mono-beam flat field at several energies on the GMT branch if available;
  - dead time: measure counts vs S3 opening on a strong spot (non-paralysable retrigger model vs vendor table); decide whether to acquire with the vendor count-rate correction ON or OFF (it is tabulated for a monochromatic beam, as is the flat field — **disable or record both** for white-beam work);
  - pulse pile-up of two photons in the same shaping window (spectral hardening at high rates);
  - pixel-to-pixel efficiency (factory flat field not valid in white beam): use the Ge fluorescence image itself (uniform, single energy ~9.9 keV) as a white-beam-compatible flat field;
  - module gaps, dead pixels, 16-bit counters / auto-summation.

### Stage 5 — Backgrounds
* Done: sample K fluorescence.
* To do: Compton (incoherent) scattering of the sample (xraylib DCSP_Compt, polarised), thermal diffuse scattering around strong spots, air scattering of the incident and diffracted beams, fluorescence of sample environment (Be dome, holder), Ir fluorescence from the KB (shielded).

### Stage 6 — Comparison with data and calibration (started, see above)
1. Acquire a dedicated Ge (001) or (111) data set on LaueMAX: several S3 openings (linearity, dead time), several exposures, threshold scan (4, 6, 8, 10, 12, 15 keV), two detector distances, monitor and ring current recorded, attenuators known.
2. ROI integration around predicted spots with local background (re-use LaueImProc/LaueTools peak integration); same ROIs in the simulation (`spot_counts`).
3. Fit: global scale k (flux x efficiency), smooth T(E) (few spline knots), charge-sharing sigma, dead time; harmonics give leverage because one spot mixes known energy ratios.
4. Cross-checks: Ge fluorescence level (threshold < and > 9.9 keV), rainbow-filter dips depth (local spectrum), Si reference crystal, pyXRT XRF-based flux.
5. Deliverable: a validated "instrument function" (spectrum x detector) file per configuration, usable by other samples.

### Stage 7 — Integration and applications
* Expose `predict_intensities(fitfile, detfile, h5)` to LaueTools (column `Itheo` next to `Itot` in `.fit`), and an option in the LaueTools simulation to return absolute intensities.
* Use cases: structure-factor/occupancy refinement on single grains, composition from intensity ratios (anomalous contrast across edges with the rainbow filter), dislocation density from integrated vs peak intensity (extinction), dose estimates (link with the dose tool of Jousseaume et al.).

## Validation checklist (automated in `tests/`)
* pixel <-> direction vs LaueTools (with and without depth offset), B0 vs `calc_B_RR`, spot positions and energies vs `calcSpots_fromHKLlist`;
* dynamical integral = Darwin (8/3) limit without absorption (sigma and pi, asymmetric);
* dynamical -> kinematic for a weak reflection;
* total flux and mean energy vs LaueMAX draft;
* detector: ~1 count per photon at threshold = E/2, > 1 at 4 keV.

## Open questions for the team
1. KB grazing angles actually used (2.8 vs 3.0 mrad): they set the high-energy end, i.e. the intensity of the hardest harmonics.
2. Is the EIGER2 count-rate correction / flat field ON during Laue acquisitions? Which threshold(s) are routinely used?
3. Sensor bias and any window/foil in front of the CdTe sensor.
4. Availability of a Ge image series with recorded S3 gaps, current, exposure (ideally with threshold scan).
5. Should `lauequant` live inside LaueTools (sub-package) or stay standalone with a thin adapter?
