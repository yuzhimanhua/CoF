# Chain-of-Factors Paper-Reviewer Matching

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

This repository contains the code, datasets, and pre-trained model used in our paper: [Chain-of-Factors Paper-Reviewer Matching](https://arxiv.org/abs/2310.14483).

## Links
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Datasets](#datasets)
- [Citation](#citation)

## Installation
We use one NVIDIA RTX A6000 GPU to run the evaluation code in our experiments. The code is written in Python 3.8. You can install the dependencies as follows.
```
conda env create --file=environment.yml --name=cof
conda activate cof
./setup.sh
```

## Quick Start
You need to first download the [**datasets**](https://drive.google.com/file/d/1IbQBVrpu3wMXahkqKVFDDYlB5_nS5cTx/view?usp=sharing) and the [**pre-trained model**](https://drive.google.com/file/d/1n4fV6-K18V1nuLPGVBTbxsU78KDCnPLd/view?usp=sharing). After you unzip the downloaded files, put the folder (i.e., ```data/``` and ```model/```) under the repository main folder ```./```.

After that, you can run our evaluation script:
```
./run.sh
```

Soft/Hard P@5 and P@10 scores will be shown at the end of the terminal output as well as in ```./scores.txt.```

## Datasets
We use four datasets - NIPS, SciRepEval, SIGIR, and KDD - in our paper. More details about each dataset are as follows.

| Dataset | #Papers | #Reviewers | #Annotated (Paper, Reviewer) Pairs | Conference(s) | Source |
| ----- | ----- | ----- | ----- | ----- | ----- |
| NIPS | 34 | 190 | 393 | NIPS 2006 | [Link](https://mimno.infosci.cornell.edu/data/nips_reviewer_data.tar.gz) |
| SciRepEval | 107 | 661 | 1,729 | NIPS 2006, ICIP 2016 | [Link](https://huggingface.co/datasets/allenai/scirepeval/viewer/paper_reviewer_matching) |
| SIGIR | 73 | 189 | 13,797 | SIGIR 2007 | [Link](https://timan.cs.illinois.edu/ir/data/ReviewData.zip) |
| KDD | 174 | 737 | 3,480 | KDD 2020 | Newly constructed by us |

## Citation
If you find our code, model, or the KDD dataset useful in your research, please cite the following paper:
```
@inproceedings{zhang2025chain,
  title={Chain-of-factors paper-reviewer matching},
  author={Zhang, Yu and Shen, Yanzhen and Kang, SeongKu and Chen, Xiusi and Jin, Bowen and Han, Jiawei},
  booktitle={WWW'25},
  year={2025}
}
```
