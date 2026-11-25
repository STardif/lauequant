# lauequant — quantitative forward model for Laue microdiffraction on BM32 (LaueMAX)

Prototype, v0.1. See `PLAN.md` for the physics, status and roadmap.

```
lauequant/
  spectrum.py    incident spectrum (wraps pyXRT), polarisation, KB convergence
  geometry.py    detector (LaueTools 5-parameter calibration) and sample geometry
  crystal.py     lattice, B0, structure factors (xraylib f0, f', f''), absorption
  scattering.py  kinematic and dynamical integrated reflectivities
  detector.py    EIGER2 CdTe photon-counting response (Monte Carlo)
  background.py  sample fluorescence on the detector
  simulate.py    LaueSimulator: spot list, per-pixel counts, images
  compare.py     read BLISS scans (pyXRT), ROI integration, model-vs-data comparison
tests/           pytest consistency checks (incl. against LaueTools)
notebooks/       01_Ge_quantitative_laue.ipynb, 02_Ge_EIGER_comparison.ipynb (+ build scripts)
data/            cached BM32 spectrum, Ge reference .fit from the LaueTools repo
```

## Install / run
```
pip install numpy scipy pandas matplotlib xraylib      # + LaueTools (optional, for cross-checks)
git clone https://github.com/STardif/pyXRT             # beamline model
export PYXRT_PATH=/path/to/pyXRT
pytest -q tests
jupyter lab notebooks/01_Ge_quantitative_laue.ipynb
```
Without pyXRT, `IncidentSpectrum.from_csv("data/bm32_spectrum_200mA_S3_0.2x0.2.csv")` gives the cached spectrum
(200 mA, S1 0.22x0.15, S3 0.2x0.2 mm, KB 3 mrad).

## Minimal use
```python
from lauequant import *
spec = IncidentSpectrum.from_pyxrt(vg3=0.2, hg3=0.2, current_a=0.2)
geo  = DetectorGeometry.from_lauetools_det("calib.det", shape=(2162, 2068))
sim  = LaueSimulator(Crystal.germanium(), UB, spec, geo, model="dynamical",
                     detector=PhotonCountingDetector(threshold_kev=4.0))
spots = sim.spots()                       # photons/s per spot, harmonics included
roi   = sim.spot_counts((1, 1, 1))["roi"] # counts/s per pixel around a spot
img   = sim.image(exposure_s=0.1, background=fluorescence_background(sim))["counts"]
```
UB is the LaueTools orientation matrix (q = UB B0 hkl, LT frame).

## Authors and licence
stardif (stardif@esrf.fr) and Claude (Anthropic). MIT licence, as LaueTools.
Notebooks are stored without outputs: run them to regenerate the figures.
