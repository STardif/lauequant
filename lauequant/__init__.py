"""lauequant: quantitative forward model of Laue microdiffraction on BM32 (LaueMAX).

Chain: incident spectrum (pyXRT)  ->  geometry (LaueTools conventions)  ->
scattering (kinematic / dynamical)  ->  photon-counting detector response.
"""
from .constants import HC_KEV_A, lt_to_lt2, M_LT_TO_LT2
from .spectrum import IncidentSpectrum, FocusedBeam, bending_magnet_sigma_fraction
from .geometry import DetectorGeometry, SampleGeometry
from .crystal import Crystal, Atom
from .detector import PhotonCountingDetector
from .simulate import LaueSimulator
from .background import fluorescence_background

__version__ = "0.1.0"

__all__ = ["IncidentSpectrum", "FocusedBeam", "DetectorGeometry", "SampleGeometry", "Crystal", "Atom",
           "PhotonCountingDetector", "LaueSimulator", "fluorescence_background", "lt_to_lt2",
           "M_LT_TO_LT2", "HC_KEV_A", "bending_magnet_sigma_fraction"]
