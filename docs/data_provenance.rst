Data and literature provenance
==============================

HIP RV series originate from the BinaryStars/Videla material. Videla et al.
(2022), DOI 10.3847/1538-3881/ac5ab4, combines astrometry and RV; this pipeline
uses RV only. Direct amplitude references are Mendez et al. (2017), DOI
10.3847/1538-3881/aa8d6f; Halbwachs et al. (2016), DOI 10.1093/mnras/stv2497;
and Tokovinin (2018), DOI 10.3847/1538-3881/aacb78, external orbit A,B.

HD170582: Mennickent et al. (2015), DOI 10.1093/mnras/stv008, donor RV.
HD50526: Rosales et al. (2021), DOI 10.3847/1538-3881/ac02ba, donor RV.
2MASS J05215658+4359220: TRES relative RV only, Thompson et al., arXiv
1806.02751v2 table 3; final publication DOI 10.1126/science.aau4005.
The numerical reference explicitly identifies the preprint version.

The HD50526 epoch HJD2454566.392 is an explicitly documented reconstruction
of an inconsistent printed offset, not an author-confirmed erratum.
The literature registry preserves source/table/PDF page, estimator, units
and interval definition. K references from different solutions are not
combined into a fictitious single published orbit.

construir_benchmark.py preserves the controlled numerical transcription.
Regenerating its PDF hashes requires supplying the original source PDFs
at the paths recorded in manifest_fuentes.json. These papers are not
distributed with the code. Reproducing numerical figures from the reference
CSV and archived chains does not require the PDFs.

Simulated data are synthetic Keplerian RV plus independent Gaussian noise.
Truth and random seeds are stored separately from fitting configuration.
These synthetic truths are not new measurements of the named stars.

All 100 simulated posterior chain sets are included in this release.
The observed archive preserves all 76 versioned configurations; only the
36 current configurations enter the final observed comparisons.
