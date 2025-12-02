# VeloTrace
VeloTrace is a Python package for RNA velocity trajectory inference.
## Installation
Conda（推荐）
GPU：conda env create -f environment.yml
CPU：删除 environment.yml 中 nvidia 条目后再创建
pip：
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt