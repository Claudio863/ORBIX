ORBIX
=====

Keplerian radial-velocity inference for spectroscopic binaries. This repository
includes a Streamlit interface, a performance study, archived numerical results
and executed notebooks. Tag v0.4.1 identifies the archived study release.

Repository layout
-----------------

::

    ORBIX/
        orbix/                    Shared Python inference and analysis
        modelos/                  Stan models and shared C++ Kepler solver
        app/                      Streamlit interface and examples
        análisis_desempeño/
            corridas_masivas/     Observations, configuration, chains, notebooks
            simulaciones/         All 100 datasets and their complete chains
            scripts/              Generation, analysis and scientific audits
            03_simulaciones_resultados.ipynb
        docs/                     Scientific documentation
        reports/                  Verification reports and manifests

Both areas import the root orbix package and use modelos. The current Streamlit
app calls ``orbix.ajuste.correr`` through ``app/servicio.py`` and uses
the same sb1_K/sb2_K models as the performance notebooks. SB2 estimates K1
and K2 directly and derives q = K1/K2 for every joint posterior sample.
The shared C++ template remains immutable: model-specific headers are prepared
in build directories. Interactive fits compile private copies of the model
and save their configuration, inputs, chains, diagnostics and provenance in
``app/resultados/`` (local, ignored by Git). The archived experiments are separate.

Interactive workflow
--------------------

Upload CSV files with ``epoch,RV,err`` columns, with velocities and uncertainties
in km/s. Declare the epoch format, choose SB1 or SB2, then choose free P/e,
fixed P, fixed e, or both fixed. Period search bounds and fixed periods are
entered in days. Priors for the remaining parameters are configurable; the
effective priors and data-derived amplitude bounds are shown before execution.
K is free for SB1; K1/K2 are free for SB2. There is no independent prior for q.

The default run uses global period initialization, 200 SB1 or 300 SB2 L-BFGS
starts, and four NUTS chains with 1000 warmup and 1000 retained iterations each.
Results include 95% HPDI intervals, orbital curves, normalized residuals and
sampling diagnostics, with a downloadable ZIP. Successful execution and
adequate residuals are distinct from satisfactory sampling diagnostics.

In the app, configure the scheme, priors and execution in the sidebar.
Select “Ejecutar optimización del sistema” to rank candidates and inspect
curves and residuals. The optimization ZIP contains the observations, settings,
L-BFGS records, candidates and run provenance. Continue with NUTS using the best saved candidate and its saved
settings. Run a new preliminary fit to apply changes to the sidebar controls.
Batch analysis is available in the performance notebooks.

For the interactive example, use ``component_1.csv`` from
``análisis_desempeño/corridas_masivas/data/HIP89000/`` for SB1. For SB2, add
``component_2.csv``. Select Besselian years, free P and e, and a search window
of 30–1000 days. The verified interface runs are summarized in
``reports/validacion_interfaz_hip89000.json``.

Installation
------------

Use Python 3.12.10. A Conda environment can be created with
``conda env create -f environment.yml`` from this root. Alternatively::

    python -m pip install -r requirements.txt
    python -m ipykernel install --user --name binarystars --display-name "binarystars - Conda"
    python -m streamlit run app/App.py

requirements-lock.txt records the exact environment. New inference requires
CmdStan 2.40.0 and a C++ toolchain. Plotting archived chains does not compile
Stan or execute new inference.

Notebooks
---------

* análisis_desempeño/corridas_masivas/01_corrida_masiva.ipynb: controlled new
  experiment entry; RUN_INFERENCE=False by default; archived runs are protected.
* análisis_desempeño/corridas_masivas/02_graficos_tablas.ipynb: observed tables,
  literature, uncertainty, diagnostics, all 36 detailed figures, correlations,
  and traces, with executed outputs retained. Standard reports retain the
  application layout: marginal distributions, pair dependence, RV versus phase
  and RV versus time. Their affine pair-axis scaling is documented explicitly.
* análisis_desempeño/03_simulaciones_resultados.ipynb: all simulated results,
  seven experimental figures and support/sampling investigations.

Verification
------------

From the repository root::

    python -m unittest discover -s análisis_desempeño/corridas_masivas/tests -v
    python -m unittest discover -s app/tests -v
    python análisis_desempeño/scripts/verificar_repositorio.py

Scientific scope
----------------

The observed study has 36 configurations; 34 satisfy strict sampling diagnostics
and 12 have reduced chi-square above three. The latter is a descriptive marker,
not a universal adequacy test. The simulated study compares initialization
against an empirical best reference, not a certified global optimum. Ten true
K2 values were outside the fitted support and 13 runs retained sampling alerts.
These groups are disjoint; coverage on 77 eligible cases is conditional and
exploratory. See docs/limitations.rst for the interpretation.

Frozen experimental metadata retains original timestamps and paths as
provenance; runtime readers resolve files within this repository. The manuscript, article PDFs and document-generation utilities are maintained
in a separate folder outside this software repository. Notebook execution
writes scientific plots and numerical tables only within the analysis area.

Attribution and citation
------------------------

CITATION.cff identifies version 0.4.1. The BSD-3-Clause software license retains
Miguel Videla's attribution and Claudio Ignacio Gómez Mayorga's contributions.
Articles, published observations and third-party illustrations retain their
original rights; the software license does not relicense those materials.
See docs/attribution.rst and the source identifiers in the numerical reference
CSV. The public repository is https://github.com/Claudio863/ORBIX .
No archival DOI is assigned here; local updates are not published automatically.
