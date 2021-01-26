#!/usr/bin/env python3

from flask import Flask, request, render_template, url_for, redirect
import torch
import numpy as np
import json
import time
import sys

CLIP_DIR = "../CLIP"
DATASET_DIR = "../some-image-dataset/"

sys.path.append(".")
import util
sys.path.append(CLIP_DIR)
import clip

device = "cuda" if torch.cuda.is_available() else "cpu"

start_time = time.time()
# image_embeddings = torch.load(DATASET_DIR+'image_embeddings.pt').to(device)
print(f"[+] Loaded image_embeddings.\t({time.time()-start_time:.3f}s)")

start_time = time.time()
model, preprocess = clip.load("ViT-B/32", device=device)
print(f"[+] Loaded model.\t\t({time.time()-start_time:.3f}s)")


app = Flask(__name__)

@app.route('/')
def index():
	return render_template("index.html")

def search_query(inp_query, embedding_db, cpu=False):
	text = clip.tokenize([inp_query])
	if not cpu:
		text = text.to(device)
	with torch.no_grad():
		text_features = model.encode_text(text)
	# normalization as done in CLIP
    # image_features = image_features / image_features.norm(dim=-1, keepdim=True)	# pre do it
    # text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # # cosine similarity as logits
    # logit_scale = self.logit_scale.exp()
    # logits_per_iamge = logit_scale * image_features @ text_features.t()
    # logits_per_text = logit_scale * text_features @ image_features.t()

	hits = util.semantic_search(text_features, embedding_db)
	return hits[0]

cache_hits = {}
# TODO: Limit cache_hits size.
def get_resp_dicts(query):
	resp = []
	hits = cache_hits.get(query)
	if hits is None:
		hits = search_query(query, image_embeddings)
		cache_hits[query] = hits
	for hit in hits:
		resp.append({
				"value": corpus_sentences[hit['corpus_id']],
				"score": f"{hit['score']:.3f}",
				"url": url_for('view', qid=hit['corpus_id'])
				})
	return resp

@app.route('/search')
def search():
	start_time = time.time()
	resp = []
	query = request.args.get('q')
	if query:
		resp = get_resp_dicts(query)
	end_time = time.time()
	time_taken = end_time-start_time
	return render_template("search.html", resp=resp, query=query, time_taken=f"{time_taken:.4f}")


@app.route('/view/<qid>')
def view(qid, view_page=True):
	start_time = time.time()
	try:
		query = corpus_sentences[int(qid)]
		resp = get_resp_dicts(query)
	except IndexError:
		query = ""
		resp = []
	end_time = time.time()
	time_taken = end_time-start_time
	return render_template("search.html", resp=resp[1:], query=query, time_taken=f"{time_taken:.4f}",
							view_page=view_page)	 # skip first resp as it is same as question in db.


	
if __name__ == '__main__':
	app.run(debug=False)
#	app.run(host='0.0.0.0', port=31796, debug=False)