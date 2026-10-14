"""campyevo: repeated thermal selection of C. jejuni in inter-/intra-strain variability space.

Companion code for "Simulating the selection of heat-resistant Campylobacter
jejuni during repeated heating, from the variability measured among and within
strains" (Microbiology Spectrum, 2026).
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"
SIMS = RESULTS / "sims"
POSTERIOR = RESULTS / "posterior"

# QUICK=1 (environment variable CAMPYEVO_QUICK) shrinks every run so that all
# notebooks finish in a few minutes; use it only to check that the code runs.
QUICK = os.environ.get("CAMPYEVO_QUICK", "0") == "1"


def n(full: int, quick: int) -> int:
    """Number of chains / iterations: `full` normally, `quick` in QUICK mode."""
    return quick if QUICK else full


for _d in (FIGURES, TABLES, SIMS, POSTERIOR):
    _d.mkdir(parents=True, exist_ok=True)

__version__ = "1.0.0"
