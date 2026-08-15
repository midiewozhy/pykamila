# KAMILA - Python Implementation

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://img.shields.io/pypi/v/pykamila.svg)](https://pypi.org/project/pykamila/)

## Overview

**KAMILA** (KAy-means for MIxed LArge data) is a semi-parametric clustering algorithm designed for datasets containing both continuous and categorical variables.

This package is a Python implementation based on:

> Foss, A., Markatou, M., Ray, B., & Heching, A. (2016).  
> *A semiparametric method for clustering mixed data.*  
> Machine Learning, 105(3), 419–458. [DOI: 10.1007/s10994-016-5575-7](https://doi.org/10.1007/s10994-016-5575-7)

**Note**: This is an experimental, work-in-progress implementation created for research purposes. Feedback and contributions are welcome!

---

## Installation

```bash
# From PyPI
pip install pykamila

# From source (development mode)
git clone [https://github.com/midiewozhy/pykamila.git](https://github.com/midiewozhy/pykamila.git)
cd pykamila
pip install -e .
```

---

## Requirements

- Python ≥ 3.8
- NumPy ≥ 1.20
- SciPy ≥ 1.7
- scikit‑learn ≥ 1.0

---

## API Reference

> class KAMILA(n_clusters, n_init=10, max_iter=300, random_state=None, con_init='default', cat_init='default')


| Parameter | Type | Description |
| --- | --- | --- |
| n_cluster | int | Number of clusters |
| con_init | str | Initialization for continuous centroids ('default' or 'kmeans++') |
| cat_init | str | Initialization for categorical parameters ('default') |
| n_init | int | Number of random initialization |
| max_iter | int | Maximum iterations per initialization |
| random_state | int/None | Random seed |

---

## Methods

> fit(X, con_idx, scales = None): Fits the model to dataset X using continuous column indices con_idx(a numpy array)

> predict(X): Predicts cluster assignments for new data X.

---

## Validation

This implementation reproduces the original paper's simulation results for p-generalized distribution (kurtosis = 6) across multiple overlap configurations. Below is part of the validation result.

| Categorical Overlap | Continuous Overlap | ARI Mean | ARI Std |
| --- | --- | --- | --- |
| 0.01 | 0.01 | 1.000 | 0.001 |
| 0.01 | 0.15 | 0.999 | 0.002 |
| 0.01 | 0.30 | 0.999 | 0.002 |
| 0.01 | 0.45 | 0.999 | 0.002 |

However, for log-normal data, the code does not generate ideal simulation results, which suggests possible improvement in the future.

---

## Citation

If you use this package, please cite the orginal paper:

```
@article{foss2016semiparametric,
  title={A semiparametric method for clustering mixed data},
  author={Foss, Alex and Markatou, Marianthi and Ray, Bonnie and Heching, Aliza},
  journal={Machine Learning},
  volume={105},
  number={3},
  pages={419--458},
  year={2016},
  publisher={Springer}
}
```

Optionally, you can also cite this Pyhong implementation:

```
@misc{pykamila,
  author = {Marshal Hong Yuan Zhu},
  title = {pykamila: A Python implementation of the KAMILA clustering algorithm},
  year = {2026},
  publisher = {GitHub},
  url = {[https://github.com/midiewozhy/pykamila](https://github.com/midiewozhy/pykamila)}
}
```

---

## License

Released under the GNU General Public License v3.0. See LICENSE for details.

---

## Contributing

Pull requests and bug reports are welcome on GitHub!

---

## Acknowledgements

Special thanks to the original authors (Alex Foss, Marianthi Markatou, Bonnie Ray, and Aliza Heching) and the authors of the original R package kamila.

---

## Contact

For questions or suggestions, please reach out via:

- Email: marshal.hy.zhu@gmail.com
- GitHub Issues: https://github.com/midiewozhy/pykamila/issues