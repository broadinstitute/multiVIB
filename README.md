multiVIB: A Unified Probabilistic Contrastive Learning Framework for Atlas-Scale Integration of Single-Cell Multi-Omics Data
=======

multiVIB is a unified framework to integrate single-cell multi-omics datasets across different scenarios.

Introduction
-----------------------
Comprehensive brain cell atlases are essential for understanding neural functions and enabling translational insights. As single-cell technologies proliferate across experimental platforms, species, and modalities, these atlases must scale accordingly, calling for integration frameworks capable of aligning heterogeneous datasets without erasing biologically meaningful variations.

Existing tools typically focus on narrow integration scenarios, forcing researchers to assemble ad hoc workflows that often introduce artifacts.
multiVIB addresses this limitation by providing a unified probabilistic contrastive learning framework that supports diverse single-cell integration tasks.

multiVIB:
Achieves state-of-the-art performance across multiple integration benchmarks
Mitigates spurious alignments by preserving biologically meaningful heterogeneity
Scales to atlas-level datasets across species and modalities
Provides a principled and extensible foundation for building next-generation brain cell atlases

Applied to datasets from the BRAIN Initiative, multiVIB demonstrates robust, scalable integration, including cross-modality and cross-species settings while preserving species-specific features.
![multiVIB](https://github.com/broadinstitute/multiVIB/blob/main/doc/figure/Figure1_framework.png?raw=false)


Navigating this Repository
--------------------------

The multiVIB repository is organized as follows:
```
<repo_root>/
├─ multiVIB/              # multiVIB python package
└─ docs/                  # Package documentation
    └─ source/
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
- **[01_basic_usage.ipynb](tutorials/01_basic_usage.ipynb)**  
  Minimal example showing data loading, model training, and latent embedding extraction.

- **[02_multimodal_integration.ipynb](tutorials/02_multimodal_integration.ipynb)**  
  Integration of multi-omics datasets (e.g., RNA + ATAC).

- **[03_cross_species_integration.ipynb](tutorials/03_cross_species_integration.ipynb)**  
  Human–mouse integration demonstrating preservation of species-specific variation.

- **[04_atlas_scale_pipeline.ipynb](tutorials/04_atlas_scale_pipeline.ipynb)**  
  Full pipeline for atlas-scale integration using large BRAIN Initiative datasets.


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
