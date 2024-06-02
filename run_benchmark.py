"""Example of a benchmark run."""
from BenchTDA import benchmark
from ExamplePipelines.fscs import pc_to_alpha
from ExamplePipelines.phs import st_to_pd
from ExamplePipelines.vecs import MeanVarBaseline, ConcatenationBaseline, PImage, PLandscape

DATA_DIR = "BenchTDA_data/"

# Set to True to rerun to check that the pipelines are deterministic
FORCE = False

FSCS = [
    # Identity function for point clouds
    {
        "in": "pc",
        "out": "pc",
        "fns": {
            "id": lambda x, seed: x
        }
    },
    # Point cloud to Gudhi SimplexTree
    {
        "in": "pc",
        "out": "gudhi_st",
        "fns": {
            "gudhi_pc_to_alpha": pc_to_alpha,
        }
    },
]

PHS = [
    # Identity function for point clouds
    {
        "in": "pc",
        "out": "pc",
        "fns": {
            "id": lambda x, seed: x
        }
    },
    # Gudhi SimplexTree to persistence diagram
    {
        "in": "gudhi_st",
        "out": "pd",
        "fns": {
            "gudhi_st_to_pd": st_to_pd,
        }
    },
]

VECS = [
    # Baseline non-TDA vectorization classes
    {
        "in": "pc",
        "classes": {
            "Concatenated points (Non-TDA)": ConcatenationBaseline,
            "Mean/var (Non-TDA)": MeanVarBaseline,
        }
    },
    # Persistence diagram-based vectorization classes
    {
        "in": "pd",
        "classes": {
            "Persistence Image": PImage,
            "Persistence Landscape": PLandscape,
        }
    },
]


benchmark(fscs=FSCS, phs=PHS, vecs=VECS, force=FORCE, verbose=True)
