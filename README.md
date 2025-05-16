# IMLE Policy

[[Project page]](https://imle-policy.github.io/)
[[Paper]](https://www.roboticsproceedings.org/rss21/p158.pdf)
[[Data]](https://huggingface.co/datasets/krishanrana/imle_policy/resolve/main/datasets.zip)
[[Colab]](#)


[Krishan Rana](https://krishanrana.github.io/)<sup>†,1</sup>,
[Robert Lee](https://scholar.google.com.au/citations?user=1Vqlm0kAAAAJ&hl=en)<sup>†</sup>,
[David Pershouse](#)<sup>1</sup>,
[Niko Suenderhauf](https://nikosuenderhauf.github.io/)<sup>1</sup>,

<sup>†</sup>Equal Contribution,
<sup>1</sup>Queensland University of Technology,

<img src="media/main.png" alt="drawing" width="100%"/>

## Installation

Download our source code:
```bash
git clone https://github.com/krishanrana/imle_policy.git
cd imle_policy
```

Create a virtual environment with Python 3.10 and activate it, e.g. with [`miniconda`](https://docs.anaconda.com/free/miniconda/index.html):
```bash
conda create -y -n imle_policy python=3.10
conda activate imle_policy
```

Install all requirements:
```bash
pip install -e .
```

Download Mujoco for the Kitchen and UR3 Block Push environments:
```bash
./get_mujoco.sh
```
Download all the required datasets and extract:
```bash
cd imle_policy
wget https://huggingface.co/datasets/krishanrana/imle_policy/resolve/main/datasets.zip && unzip datasets.zip -d datasets && rm datasets.zip
```

## Quick Start

To train IMLE Policy on the PushT task with all the default parameters, run:
```bash
python train.py --task pusht --method rs_imle 
```

Available options:

> **task:** pusht, Lift, NutAssemblySquare, PickPlaceCan, ToolHang, TwoArmTransport, kitchen, ur3_blockpush

> **method:** rs_imle, diffusion, flow_matching

> **dataset_percentage:** Fixed subsample of the full dataset ranging from 0.1 to 1.0

> **epsilon:** IMLE Policy-specific hyperparameter that controls the rejection sampling threshold

> **n_samples_per_condition:** IMLE Policy-specific hyperparameter that controls the number of samples per condition

> **use_traj_consistency:** IMLE Policy-specific hyperparameter that controls whether to use trajectory consistency or not
