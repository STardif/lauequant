# data

* `bm32_spectrum_200mA_S3_0.2x0.2.csv` — incident spectrum at the sample computed with pyXRT
  `BM32_WHITE_BEAM(vg1=0.22, hg1=0.15, vg3=0.2, hg3=0.2, current_a=0.2)`, KB at 3 mrad
  (columns: energy keV, photons/s/eV, sigma-polarised fraction). Fallback when pyXRT is not installed.
* `Ge_blanc_0000_LT_5.fit` — indexed Ge pattern (MarCCD, pre-EBS) from the LaueTools repository
  (`LaueTools/LaueImages`, MIT licence), used in notebook 01 §8.

LaueMAX measurements (BLISS master + EIGER2 images) are not stored here; notebook 02 reads them from
`$GE_DATA` (master file `master.h5`, `calib.det`, and `scan0001/eiger4m_0000.h5` next to the master).
