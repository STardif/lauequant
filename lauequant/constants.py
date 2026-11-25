"""Physical constants and frame conventions used across lauequant.

Frames
------
LaueTools uses two laboratory frames (see ``LaueTools.LaueGeometry``):

* **LT frame** (orientation matrices, ``q = UB B0 G*``): x // incident beam,
  z towards the detector in top-reflection geometry (vertical, up),
  y = z x x (horizontal, towards the hutch door).
* **LT2 frame** (detector geometry, ``calc_xycam`` / ``calc_uflab``): y // incident
  beam, z vertical (up), x = y x z ... i.e. horizontal towards the wall.
  ``x_LT2 = -y_LT``, ``y_LT2 = x_LT``, ``z_LT2 = z_LT``.

All geometry in lauequant is computed in the **LT2 frame**; orientation
matrices read from LaueTools files are converted with :func:`lt_to_lt2`.
Reciprocal vectors are in 1/Angstrom **without** the 2*pi factor, as in LaueTools.
"""
import numpy as np

#: hc in keV * Angstrom (CODATA). LaueTools uses 12.398; the difference (3e-5)
#: is irrelevant for intensities but shows in 5th digit of spot energies.
HC_KEV_A = 12.398419843320026
HC_LAUETOOLS = 12.398

#: Classical electron radius in Angstrom.
R_E_A = 2.8179403262e-5

#: Avogadro number.
N_A = 6.02214076e23

#: Air composition (mass fractions) and density at 20 C, 1013 hPa (g/cm3).
AIR_MASS_FRACTIONS = {"N": 0.7553, "O": 0.2318, "Ar": 0.0128, "C": 0.0001}
AIR_DENSITY = 1.204e-3

#: Incident beam direction in LT2.
KI_LT2 = np.array([0.0, 1.0, 0.0])


def lt_to_lt2(v):
    """Convert vectors (..., 3) or a 3x3 matrix acting on LT vectors from LT to LT2.

    For a vector: (x, y, z)_LT -> (-y, x, z)_LT2.
    """
    v = np.asarray(v, dtype=float)
    return v[..., [1, 0, 2]] * np.array([-1.0, 1.0, 1.0])


#: Rotation matrix M such that v_LT2 = M @ v_LT.
M_LT_TO_LT2 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def lt2_to_lt(v):
    """Inverse of :func:`lt_to_lt2`."""
    v = np.asarray(v, dtype=float)
    return v @ M_LT_TO_LT2  # (M^T v) for row vectors


def normalize(v, axis=-1):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v, axis=axis, keepdims=True)
