# MBL_CellSegmentation

## Getting Started

Create a new environment
```bash
conda create -n mbl_cellsegmentation python=3.12
conda activate mbl_cellsegmentation
conda install pytorch pytorch-cuda=12.1 boost psycopg2 -c pytorch -c nvidia -y
conda install -c conda-forge -c ostrokach-forge -c pkgw-forge graph-tool -y
pip install -e .
```
