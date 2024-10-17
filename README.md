# Chain-of-Factors Paper-Reviewer Matching

This repository contains the code, datasets, and pre-trained model related to our paper: _Chain-of-Factors Paper-Reviewer Matching_.

## Datasets
We use four datasets - NIPS, SciRepEval, SIGIR, and KDD - in our paper. They can be downloaded [**here**](https://drive.google.com/file/d/19_9ZBTZvemP_V6s1PpMt6H-XxfX_sXvH/view?usp=sharing). After you unzip the file, put the folder (i.e., ```data/```) under the repository main folder ```./```.

## Model
The pre-trained Chain-of-Factors model can be downloaded [**here**](https://drive.google.com/file/d/1pT6UX3TTy7WOzTsBDkpWZtGQZPWm0kwp/view?usp=sharing). After you unzip the file, put the folder (i.e., ```model/```) under the repository main folder ```./```.

## Running Evaluation Code
GPUs are required. We use one NVIDIA RTX A6000 GPU in our experiments. The code is written in Python 3.8. You can install the dependencies by running:
```
conda env create --file=environment.yml --name=cof
conda activate cof
./setup.sh
```

After that, you can run our evaluation script:
```
./run.sh
```

Soft/Hard P@5 and P@10 scores will be shown in the last several lines of the output as well as in ```./scores.txt.```
