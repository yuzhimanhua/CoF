# Chain-of-Factors Paper-Reviewer Matching

This repository contains the code, datasets, and pre-trained model related to our KDD submission: _Chain-of-Factors Paper-Reviewer Matching_.

## Datasets
We use four datasets - NIPS, SciRepEval, SIGIR, and KDD - in our paper. They can be downloaded [**here**](https://gofile.io/d/Bn5WT1).

## Model
The pre-trained Chain-of-Factors model can be downloaded [**here**](https://gofile.io/d/zMW7st).

## Running Evaluation Code
GPUs are required. We use one NVIDIA RTX A6000 GPU in our experiments. The code is written in Python 3.8. You can install the dependencies by running:
```
./setup.sh
```
Then, please unzip the downloaded dataset and model folders, and put them under ```./```. 

After that, you can run our evaluation script:
```
./run.sh
```

Soft/Hard P@5 and P@10 scores will be shown in the last several lines of the output as well as in ```./scores.txt.```
