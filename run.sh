dataset=SIGIR
model=cof.ckpt

factor=semantic
python3.8 get_paper_emb.py --dataset ${dataset} --model ${model} --factor ${factor}

factor=topic
python3.8 get_paper_emb.py --dataset ${dataset} --model ${model} --factor ${factor}

factor=citation
python3.8 get_paper_emb.py --dataset ${dataset} --model ${model} --factor ${factor}

python3.8 chain_of_factors.py --dataset ${dataset}
