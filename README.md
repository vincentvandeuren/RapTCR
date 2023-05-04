# RapTCR: Rapid TCR repertoire visualization and annotation

The acquisition of T-cell receptor (TCR) repertoire sequence data has become faster and cheaper due to advancements in high-throughput sequencing. However, fully exploiting the diagnostic and clinical potential within these TCR repertoires requires a thorough understanding of the inherent repertoire structure. Hence, visualizing the full space of TCR sequences could be a key step towards enabling exploratory analysis of TCR repertoire, driving their enhanced interrogation. Nonetheless, current methods remain limited to rough profiling of TCR V and J gene distributions. Addressing this need, we developed RapTCR, a tool for rapid visualization and post-analysis of TCR repertoires. 

To overcome computational complexity, RapTCR introduces a novel, simple embedding strategy that represents TCR amino acid sequences as short vectors while retaining their pairwise alignment similarity. RapTCR then applies efficient algorithms for indexing these vectors and constructing their nearest neighbor network. It provides multiple visualization options to map and interactively explore this TCR network as a two-dimensional representation. Benchmarking analyses using epitope-annotated datasets demonstrate that these visualizations capture TCR similarity features both globally (e.g., J gene) and locally (e.g., epitope reactivity). RapTCR is available as a Python package, implementing the intuitive scikit-learn syntax to easily generate insightful, publication-ready figures for TCR repertoires of any size.

## Documentation

Documentation is available at: https://vincentvandeuren.github.io/RapTCR_docs/
