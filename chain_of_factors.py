from tqdm import tqdm
import argparse
import numpy as np
import json
from collections import defaultdict
import pytrec_eval

parser = argparse.ArgumentParser(description='main', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', default='SIGIR')
parser.add_argument('--topn1', type=int, default=50)
parser.add_argument('--topn2', type=int, default=100)
parser.add_argument('--topk', type=int, default=3)
args = parser.parse_args()
dataset = args.dataset
topk = args.topk

paper2emb_R = {}
paper2emb_L = {}
paper2emb_C = {}
with open(f'data/{dataset}_papers_test.json') as fin1, \
	 open(f'embedding/{dataset}_paper_emb_semantic.txt') as fin2, \
	 open(f'embedding/{dataset}_paper_emb_topic.txt') as fin3, \
	 open(f'embedding/{dataset}_paper_emb_citation.txt') as fin4:
	print('Getting paper embeddings...')
	for idx, (line1, line2, line3, line4) in enumerate(tqdm(zip(fin1, fin2, fin3, fin4))):
		data1 = json.loads(line1)
		paper = data1['paper']

		data2 = line2.strip().split()
		emb_R = np.array([float(x) for x in data2])
		paper2emb_R[paper] = emb_R

		data3 = line3.strip().split()
		emb_L = np.array([float(x) for x in data3])
		paper2emb_L[paper] = emb_L

		data4 = line4.strip().split()
		emb_C = np.array([float(x) for x in data4])
		paper2emb_C[paper] = emb_C

reviewer2papers = {}
paper2reviewers = defaultdict(set)
with open(f'data/{dataset}_reviewers_test.json') as fin:
	print('Getting reviewer profiles...')
	for line in tqdm(fin):
		data = json.loads(line)
		reviewer = data['reviewer']
		papers = data['papers']
		reviewer2papers[reviewer] = papers
		for paper in papers:
			paper2reviewers[paper].add(reviewer)

if args.topn1 > 0:
	topn1 = int(len(paper2reviewers)/args.topn1)
else:
	topn1 = 1
if args.topn2 > 0:
	topn2 = int(len(paper2reviewers)/args.topn2)
else:
	topn2 = 1

for task in ['soft', 'hard']:
	qrel = {}
	run = {}
	with open(f'data/{dataset}_queries_test_{task}.json') as fin:
		print('Evaluate the {} setting...'.format(task))
		for line in tqdm(fin):
			data = json.loads(line)
			query = data['query_id']
			q_emb_R = paper2emb_R[query]
			q_emb_C = paper2emb_C[query]
			q_emb_L = paper2emb_L[query]

			p_score = {}
			for paper in paper2reviewers:
				p_emb_R = paper2emb_R[paper]
				p_score[paper] = np.dot(q_emb_R, p_emb_R)
			p_score_sorted = sorted(p_score.items(), key=lambda x:x[1], reverse=True)[:topn1]

			candidates = [x[0] for x in p_score_sorted]
			p_score = {}
			for paper in candidates:
				p_emb_C = paper2emb_C[paper]
				p_score[paper] = np.dot(q_emb_C, p_emb_C)
			p_score_sorted = sorted(p_score.items(), key=lambda x:x[1], reverse=True)[:topn2]

			reviewer2scores = defaultdict(list)
			for k, v in p_score_sorted:
				for reviewer in paper2reviewers[k]:
					p_emb_R = paper2emb_R[k]
					p_emb_C = paper2emb_C[k]
					p_emb_L = paper2emb_L[k]
					score = np.dot(q_emb_R, p_emb_R) + np.dot(q_emb_C, p_emb_C) + np.dot(q_emb_L, p_emb_L)
					# score = np.dot(q_emb_L, p_emb_L)
					reviewer2scores[reviewer].append(score)

			y = data['score']
			y_pred = {}
			for reviewer in y:
				y_pred[reviewer] = sum(sorted(reviewer2scores[reviewer], reverse=True)[:topk])/topk
			qrel[query] = y
			run[query] = y_pred

	evaluator = pytrec_eval.RelevanceEvaluator(qrel, {'P_5', 'P_10', 'map', 'ndcg'})
	results = evaluator.evaluate(run)
	p5 = pytrec_eval.compute_aggregated_measure('P', [query_measures['P_5'] for query_measures in results.values()])
	p10 = pytrec_eval.compute_aggregated_measure('P', [query_measures['P_10'] for query_measures in results.values()])
	mapr = pytrec_eval.compute_aggregated_measure('map', [query_measures['map'] for query_measures in results.values()])
	ndcg = pytrec_eval.compute_aggregated_measure('ndcg', [query_measures['ndcg'] for query_measures in results.values()])
	p5, p10, mapr, ndcg = p5*100, p10*100, mapr*100, ndcg*100

	print('[Result]', task, 'P@5:', p5)
	print('[Result]', task, 'P@10:', p10)
	with open('scores.txt', 'a') as fout:
		if task == 'soft':
			fout.write('dataset: {}'.format(dataset)+'\n')
		fout.write('{} P@5: {:.2f}'.format(task, p5)+'\n')
		fout.write('{} P@10: {:.2f}'.format(task, p10)+'\n')
		if task == 'hard':
			fout.write('\n')
