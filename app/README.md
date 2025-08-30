# ShEPhERD Streamlit App

Interactive web interface for **ShEPhERD** (Shape, Electrostatics, and Pharmacophores for Enhanced Rational Design) - a diffusion model for bioisosteric drug design.

## Features
- **Molecule Input**: SMILES, MolBlock, XYZ, or test molecules from MOSES dataset
- **Conditional Generation**: Generate molecules matching reference shape, electrostatics, and/or pharmacophores
- **Multiple Models**: MOSES-aq (full conditioning) and GDB variants (shape/electrostatics only)
- **Interactive Visualization**: 2D/3D molecular structures with interaction profiles
- **Atom Inpainting**: Advanced control for modifying specific molecular regions to enable scaffold decoration
- **Evaluation & Export**: Similarity scoring and SDF/XYZ download of generated molecules

### Installation
```
pip install streamlit stmol "shepherd-score>=1.1.1" py3Dmol
```
NOTE: requires shepherd-score >= 1.1.1 for visualizations

### How to use
```
streamlit run app.py
```