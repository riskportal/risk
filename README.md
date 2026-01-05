# RISK

![Python](https://img.shields.io/badge/python-3.8%2B-yellow)
[![pypiv](https://img.shields.io/pypi/v/risk-network.svg)](https://pypi.python.org/pypi/risk-network)
![License](https://img.shields.io/badge/license-GPLv3-purple)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17257417.svg)](https://doi.org/10.5281/zenodo.17257417)
![Tests](https://github.com/riskportal/risk/actions/workflows/ci.yml/badge.svg)

**RISK** (Regional Inference of Significant Kinships) is a next-generation tool for biological network annotation and visualization. It integrates community detection algorithms, rigorous overrepresentation analysis, and a modular framework for diverse network types. RISK identifies biologically coherent relationships within networks and generates publication-ready visualizations, making it a useful tool for biological and interdisciplinary network analysis.

For a full description of RISK and its applications, see:
<br>
**Horecka, I. and Röst, H. (2026)**.
_RISK: a next-generation tool for biological network annotation and visualization_.
**Bioinformatics**.
https://doi.org/10.1093/bioinformatics/btaf669

## Documentation and Tutorial

Full documentation is available at:

- **Docs:** [https://riskportal.github.io/risk-docs](https://riskportal.github.io/risk-docs)
- **Tutorial Jupyter Notebook Repository:** [https://github.com/riskportal/risk-docs](https://github.com/riskportal/risk-docs)

## Installation

RISK is compatible with Python 3.8 or later and runs on all major operating systems. To install the latest version of RISK, run:

```bash
pip install risk-network --upgrade
```

## Key Features of RISK

- **Broad Data Compatibility**: Accepts multiple network formats (Cytoscape, Cytoscape JSON, GPickle, NetworkX) and user-provided annotations formatted as term–to–gene membership tables (JSON, CSV, TSV, Excel, Python dictionaries).
- **Flexible Clustering**: Offers Louvain, Leiden, Markov Clustering, Greedy Modularity, Label Propagation, Spinglass, and Walktrap, with user-defined resolution parameters to detect both coarse and fine-grained modules.
- **Statistical Testing**: Provides permutation, hypergeometric, chi-squared, and binomial tests, balancing statistical rigor with speed.
- **High-Resolution Visualization**: Generates publication-ready figures with customizable node/edge properties, contour overlays, and export to SVG, PNG, or PDF.

## Example Usage

We applied RISK to a _Saccharomyces cerevisiae_ protein–protein interaction (PPI) network (Michaelis _et al_., 2023; 3,839 proteins, 30,955 interactions). RISK identified compact, functional modules overrepresented in Gene Ontology Biological Process (GO BP) terms (Ashburner _et al_., 2000), revealing biological organization including ribosomal assembly, mitochondrial organization, and RNA polymerase activity.

[![RISK analysis of the yeast PPI network](https://i.imgur.com/TSxaxHl.jpeg)](https://i.imgur.com/TSxaxHl.jpeg)
**RISK workflow overview and analysis of the yeast PPI network**. Clusters are color-coded by enriched GO Biological Process terms (p < 0.01).

## Citation

### Primary citation

**Horecka, I. and Röst, H. (2026)**
_RISK: a next-generation tool for biological network annotation and visualization_.
**Bioinformatics**.
https://doi.org/10.1093/bioinformatics/btaf669

### Software archive

RISK software archive corresponding to the published manuscript.
Zenodo.
https://doi.org/10.5281/zenodo.17257418

## Contributing

We welcome contributions from the community:

- [Issues Tracker](https://github.com/riskportal/risk/issues)
- [Source Code](https://github.com/riskportal/risk/tree/main/src/risk)

## Support

If you encounter issues or have suggestions for new features, please use the [Issues Tracker](https://github.com/riskportal/risk/issues) on GitHub.

## License

RISK is open source under the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html).
