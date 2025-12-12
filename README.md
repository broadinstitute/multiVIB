multiVIB: A Unified Probabilistic Contrastive Learning Framework for Atlas-Scale Integration of Single-Cell Multi-Omics Data
=======

multiVIB is a unified framework to integrate single-cell multi-omics datasets across different scenarios.
![multiVIB](https://github.com/broadinstitute/multiVIB/blob/main/doc/figure/Figure1_framework.png?raw=false)

Introduction
-----------------------
Comprehensive brain cell atlases are essential for understanding neural functions and enabling translational insights. As single-cell technologies proliferate across experimental platforms, species, and modalities, these atlases must scale accordingly, calling for integration frameworks capable of aligning heterogeneous datasets without erasing biologically meaningful variations.

Existing tools typically focus on narrow integration scenarios, forcing researchers to assemble ad hoc workflows that often introduce artifacts.
multiVIB addresses this limitation by providing a unified probabilistic contrastive learning framework that supports diverse single-cell integration tasks.
![multiVIB](https://github.com/broadinstitute/multiVIB/blob/main/doc/figure/Figure1_scenarios.png?raw=false)


Navigating this Repository
--------------------------

The multiVIB repository is organized as follows:
```
<repo_root>/
├─ multiVIB/              # multiVIB python package
└─ doc/                   # Package documentation
    └─ tutorial/
        └─ notebooks/     # Example jupyter notebooks
```

Installation
------------
We suggest creating a new conda environment to run multiVIB

```
conda create -n multiVIB python=3.10
conda activate multiVIB

git clone https://github.com/broadinstitute/multiVIB.git
cd multiVIB

pip install .
```

Tutorial
------------
We provide end-to-end Jupyter notebooks demonstrating how to use **multiVIB** across common integration tasks.

### **Available Tutorials**
- **[01_Figure2.ipynb](doc/tutorial/01_Figur2.ipynb)**  
  Apply multiVIB to the conceptual experiment we set up in Figure 2 of our manuscript.

- **[02_multimodal_integration.ipynb](doc/tutorial/02_multimodal_integration.ipynb)**  
  Integration of multi-omics datasets (e.g., RNA + ATAC).

- **[03_cross_species_integration.ipynb](doc/tutorial/03_cross_species_integration.ipynb)**  
  Cross-species integration of mammalian basal ganlia datasets demonstrating preservation of species-specific variation.


Preprint and Citation
---------------------

If you use multiVIB in your research, please cite our preprint:

Yang Xu, Stephen Jordan Fleming, Brice Wang, Erin G Schoenbeck, Mehrtash Babadi, Bing-Xing Huo. multiVIB: A Unified Probabilistic Contrastive Learning Framework for Atlas-Scale Integration of Single-Cell Multi-Omics Data.

@article{xu2025multivib,
  title={multiVIB: A unified probabilistic contrastive learning framework for atlas-scale integration of single-cell multi-omics data},
  author={Xu, Yang and Fleming, Stephen Jordan and Wang, Brice and Schoenbeck, Erin G and Babadi, Mehrtash and Huo, Bing-Xing},
  journal={bioRxiv},
  pages={2025--11},
  year={2025},
  publisher={Cold Spring Harbor Laboratory}
}
