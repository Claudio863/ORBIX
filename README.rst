ORBIX
=====

Keplerian radial-velocity inference for spectroscopic binaries. Release 0.4.1
includes a Streamlit interface, an independent performance study, all observed
and simulated numerical results, and executed notebooks.

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

Both areas import the root orbix package and use modelos. The inherited
interface retains its historical statistical parameterization; relocation does
not establish equivalence to the validated thesis pipeline (sb1_K/sb2_K).
The shared C++ template remains immutable: model-specific headers are prepared
in build directories. Mathematical changes are not part of this reorganization.

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
  and traces, with executed outputs retained.
* análisis_desempeño/03_simulaciones_resultados.ipynb: all simulated results,
  seven experimental figures and support/sampling investigations.

Verification
------------

From the repository root::

    python -m unittest discover -s análisis_desempeño/corridas_masivas/tests -v
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
CSV. No public GitHub URL
or archival DOI has been assigned by preparing this local repository.
