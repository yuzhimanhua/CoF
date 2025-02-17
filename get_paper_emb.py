import os
from tqdm import tqdm
import argparse
import json
import torch

from dpr.models.hf_models import HFEncoder, get_any_tokenizer
from dpr.models.biencoder_uni import InstructBiEncoderUni
from dpr.utils.model_utils import move_to_device

from transformers import logging
logging.set_verbosity_error()

def get_any_biencoder_component_for_infer(cfg, use_instruct=True, **kwargs):
	dropout = 0.0
	use_vat = False
	use_moe = cfg['use_moe']
	num_expert = cfg['num_expert'] if use_moe else 0
	use_infer_expert = cfg['use_infer_expert'] if use_moe else False
	per_layer_gating = cfg['per_layer_gating'] if use_moe else False
	moe_type = cfg['moe_type'] if use_moe else None
	num_q_expert = 1

	mean_pool_q_encoder = False
	if cfg['mean_pool']:
		mean_pool_q_encoder = True

	factor_rep = cfg['factor_rep']

	if cfg["pretrained_model_cfg"] == 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract':
		cfg["pretrained_model_cfg"] = 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract'

	question_encoder = HFEncoder(
		cfg['pretrained_model_cfg'],
		projection_dim=cfg['projection_dim'],
		dropout=dropout,
		use_vat=use_vat,
		use_moe=use_moe,
		moe_type=moe_type,
		use_infer_expert=use_infer_expert,
		per_layer_gating=per_layer_gating,
		num_expert=num_expert if cfg['shared_encoder'] else num_q_expert,
		mean_pool=mean_pool_q_encoder,
		factor_rep=factor_rep,
		pretrained=True,
		use_norm_rep=False,
		task_header_type=None,
		use_instruct=use_instruct,
		use_attn_gate=cfg['use_attn_gate'] if 'use_attn_gate' in cfg else False,
		instruct_type=cfg['instruct_type'] if 'instruct_type' in cfg else 'all',
		proj_adaptor=cfg['proj_adaptor'] if 'proj_adaptor' in cfg else False,
		**kwargs
	)

	ctx_encoder = question_encoder
	instruct_encoder = question_encoder
	biencoder = InstructBiEncoderUni(
		question_model=question_encoder,
		ctx_model=ctx_encoder,
		instruct_model=instruct_encoder,
		fix_ctx_encoder=False,
		fix_instruct_encoder=False,
		q_rep_method=cfg['q_rep_method'],
	)
	tensorizer = get_any_tokenizer(cfg['pretrained_model_cfg'])

	return tensorizer, biencoder


def embed_text_psg(text_psg,
				   tokenizer,
				   embed_model,
				   instruct_hidden_states=None,
				   instruct_attention_mask=None,
				   output_hidden_states=False,
				   norm_rep=False,
				   max_len=512,
				   expert_id=None,
				   device='cuda:0'):
	inputs = tokenizer(
		text_psg,
		add_special_tokens=True,
		max_length=max_len,
		padding='max_length',
		truncation=True,
		return_offsets_mapping=False,
		return_tensors='pt',
	)
	model_inputs = {
		'input_ids': move_to_device(inputs.input_ids, device=device),
		'token_type_ids': None,
		'attention_mask': move_to_device(inputs.attention_mask, device=device),
		'output_hidden_states': output_hidden_states,
	}

	if expert_id is not None:
		model_inputs['expert_id'] = expert_id

	if instruct_hidden_states is not None:
		model_inputs['instruct_hidden_states'] = move_to_device(instruct_hidden_states, device=device)
		model_inputs['instruct_attention_mask'] = move_to_device(instruct_attention_mask, device=device)

	outputs = embed_model(**model_inputs)
	if norm_rep:
		return torch.nn.functional.normalize(outputs[1], p=2, dim=-1).cpu()
	return outputs[1].cpu()


parser = argparse.ArgumentParser(description='main', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', required=True)
parser.add_argument('--model', required=True)
parser.add_argument('--factor', required=True)
args = parser.parse_args()

# Specificies the device used for inference. Modifies it if necessary.
device = 'cuda:0'
dataset = args.dataset

# Instruction settings
use_instruct = True
use_moe = False
factor = args.factor

if dataset == 'NIPS':
	if factor == 'topic':
		query_instruct_text = 'Retrieve a scientific paper that shares similar scientific topic classes with the query.'
	elif factor == 'citation':
		query_instruct_text = 'Retrieve a scientific paper that is cited by the query.'
	elif factor == 'semantic':
		query_instruct_text = 'Retrieve a scientific paper that is relevant to the query.'
elif dataset == 'SciRepEval' or dataset == 'SIGIR':
	if factor == 'topic':
		query_instruct_text = 'Find a pair of papers that one paper shares similar scientific topic classes with the other paper.'
	elif factor == 'citation':
		query_instruct_text = 'Find a pair of papers that one paper cites the other paper.'
	elif factor == 'semantic':
		query_instruct_text = 'Find a pair of papers that one paper is relevant to the other paper.'
elif dataset == 'KDD':
	if factor == 'topic':
		query_instruct_text = 'Retrieve a scientific paper that is topically relevant to the query scientific paper.'
	elif factor == 'citation':
		query_instruct_text = 'Retrieve a scientific paper that is most likely cited by the query scientific paper.'
	elif factor == 'semantic':
		query_instruct_text = 'Retrieve a scientific paper that is semantically relevant to the query scientific paper.'

ctx_instruct_text = query_instruct_text

if use_instruct:
	if query_instruct_text is None:
		raise ValueError('When use_instruct=True, query_instruct_text is required!')

# Reads in the model parameters
model_fn = f'./model/{args.model}'
state_dict = torch.load(model_fn)
tokenizer, biencoder = get_any_biencoder_component_for_infer(state_dict['encoder_params'], use_instruct=use_instruct)
print('[Instruction]', query_instruct_text)

# CLS or mean pooling
norm_rep = False
if state_dict['encoder_params']['mean_pool']:
	norm_rep = True

# Loads the pretrain checkpoints.
biencoder.load_state_dict(state_dict['model_dict'])

# If using GPU for inference.
biencoder.to(device)
biencoder.eval()

# Instruction input
if use_instruct:
	query_instruct_inputs = tokenizer(
		query_instruct_text,
		add_special_tokens=True,
		max_length=32,
		padding='max_length',
		truncation=True,
		return_offsets_mapping=False,
		return_tensors='pt',
	)
	instruct_model_inputs = {
		'input_ids': move_to_device(query_instruct_inputs.input_ids, device=device),
		'token_type_ids': None,
		'attention_mask': move_to_device(query_instruct_inputs.attention_mask, device=device),
		'output_hidden_states': True,
	}
	query_instruct_hidden_states = biencoder.instruct_model(**instruct_model_inputs)[2]
	query_instruct_attention_mask = query_instruct_inputs.attention_mask

	ctx_instruct_hidden_states = query_instruct_hidden_states
	ctx_instruct_attention_mask = query_instruct_attention_mask


# Reads in papers.
paper_fn = f'./data/{dataset}_papers_test.json'

def paper_formatting(datum, tokenizer):
	return ' '.join([datum['title'], tokenizer.sep_token, datum['abstract']])

with open(paper_fn) as fin:
	paper_ids = [json.loads(line)['paper'] for line in fin]
	
with open(paper_fn) as fin:
	paper_texts = [paper_formatting(json.loads(line), tokenizer) for line in fin]

# Starts embedding papers.
total_data_size = len(paper_ids)
batch_size = 64
end_idx = 0
paper_embeds = []
with torch.no_grad():
	for start_idx in tqdm(range(0, total_data_size, batch_size)):
		end_idx = start_idx + batch_size
		paper_embeds.append(embed_text_psg(paper_texts[start_idx:end_idx], tokenizer, biencoder.ctx_model, \
				  device=device, norm_rep=norm_rep, expert_id=None, \
				  instruct_hidden_states=query_instruct_hidden_states, instruct_attention_mask=query_instruct_attention_mask))
		
	if end_idx < total_data_size:
		paper_embeds.append(embed_text_psg(paper_texts[end_idx:], tokenizer, biencoder.ctx_model, \
				  device=device, norm_rep=norm_rep, expert_id=None, \
				  instruct_hidden_states=query_instruct_hidden_states, instruct_attention_mask=query_instruct_attention_mask))
		
paper_tensor = torch.cat(paper_embeds, dim=0)
if not os.path.exists('./embedding/'):
	os.makedirs('./embedding/')
with open(f'./embedding/{dataset}_paper_emb_{factor}.txt', 'w') as fout:
	for emb in tqdm(paper_tensor):
		emb_str = [str(round(x.item(), 5)) for x in emb.cpu()]
		fout.write(' '.join(emb_str)+'\n')
