Methodology and reproducibility
==============================

The RV model is gamma + s K [cos(omega+nu)+e cos(omega)], with s=+1 for
the primary and -1 for the secondary. K1 and K2 are free amplitudes, while
q is derived. Times retain their declared source convention; changing an
offset is not a heliocentric/barycentric correction.

The production run uses 200 SB1 or 300 SB2 starts, 4000 maximum optimizer
iterations, 4 NUTS chains, 1000 warmup and 1000 retained draws per chain,
adapt_delta=0.95 and max_treedepth=15. Final observed runs impose a declared
60 s timeout per optimization attempt and 1800 s per sampling chain.
Four already-corrected observed runs are reused and explicitly identified.
Data, priors and fixed literature values are frozen for this comparison.

The 100-system experiment uses two strategies and two search seeds per
dataset. The empirical reference is not a certified global optimum.
There is no independent periodogram validation in this release. The
investigative audit does not rerun or repair these simulations.

Posterior summaries report median and 95% HPDI. Circular variables are
unwrapped; periastron epochs are aligned sample by sample with T+nP,
using a constant integer n for each run. Fixed parameters are conditions,
not exact measurements with inferred zero uncertainty. Literature intervals
retain their published definition; shared input data preclude an independent
validation claim. SHA-256 manifests identify analysis inputs.
