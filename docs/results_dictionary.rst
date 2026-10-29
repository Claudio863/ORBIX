Result fields and interpretation
================================

estado=ok
  The run completed and could be summarized; not a scientific validity flag.
n_inicios_ok
  Number of finite optimizer candidates. Optimization uses
  require_converged=False; this is not the number of certified normal terminations.
chi2_red_lbfgs
  Nominal reduced chi-square at the best retained optimizer candidate.
chi2_red
  Nominal reduced chi-square at the highest-log-posterior archived sample.
  This point is an empirical sample maximum, not an exact posterior MAP.
treedepth_max
  Historical field name for the number of transitions saturating the configured
  maximum depth. It is not the maximum numerical depth actually observed.
convergencia_ok
  Legacy pipeline flag for R-hat and divergences. It does not enforce all
  ESS, energy, depth and support criteria. Use evaluacion_estricta.csv for
  the complete prespecified diagnostic classification.
chi2_sobre_N
  Descriptive per-component mean squared normalized residual. Shared fitted
  parameters prevent assigning component degrees of freedom independently.
fijo
  A conditioned parameter. Its uncertainty is not inferred; it must not be
  presented as a measured zero-width uncertainty.
origen_ejecucion
  Distinguishes the four reused bounded-phase runs from 32 new observed runs.

Historical and current chains remain separate. A failed current run is not
silently replaced with an older chain. Warnings, large residuals and timeouts
are retained as research outcomes.
Unavailable numeric metadata are represented as JSON null, rather than
non-standard NaN tokens, in this public copy. Numerical chain CSVs are
preserved byte for byte.
