Known limitations
=================

The 100-system experiment completed all computations, but 10 SB2 eccentric
systems without periastron coverage exclude the true K2 from the fitted
support. Kmax=max(2*observed RV range,1) is data-dependent and can truncate
the truth even without noise. This is a model-support limitation that
cannot be repaired by improving the optimizer within that support.

Thirteen cases fail at least one prespecified sampling diagnostic. These
are not automatically optimization failures: all attained the empirical
reference with the global strategy. There are 10 R-hat warnings, 10 bulk
ESS warnings, 5 tail ESS warnings and 3 cases with divergences (overlapping
counts). Only 77 cases meet both support and strict diagnostic criteria;
their coverage is conditional and exploratory, not general calibration.

The best retained candidates of both initialization strategies terminate
normally in all 100 systems, yet the local strategy misses the empirical
reference in 24. Numerical stopping criteria do not identify the best known
mode. This comparison does not isolate individual preprocessing components
and is not an independent periodogram validation.

Historical chains used an unbounded periodic phase and are retained solely
for provenance. They must not be interpreted as validated posterior samples.
Short temporal baselines, absent periastron coverage, eccentricity/amplitude
correlation and fixed rounded periods can limit observed fits. The model
does not include instrument offsets, jitter or non-Keplerian variability.
Residual excess alone does not establish its physical cause.

Future work should predefine proper, physically justified amplitude priors
independent of observed RV extrema; assess support sensitivity; obtain
informative orbital-phase coverage; and predefine sampling extensions and
replication. This release preserves all failures instead of tuning settings
retrospectively to pass diagnostic thresholds.
