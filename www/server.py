#!/usr/bin/env python3

from flask import Flask, request, render_template, url_for, redirect
import torch
import numpy as np
import json
import time
import sys
import csv
import pickle

CLIP_DIR = "../CLIP"
DATASET_DIR = "../unsplash-image-dataset/"

sys.path.append(CLIP_DIR)
import clip

device = "cuda" if torch.cuda.is_available() else "cpu"

start_time = time.time()
images_embeddings = torch.load(DATASET_DIR+'images_embeddings.pt').to(device)
print(f"[+] Loaded images_embeddings.\t({time.time()-start_time:.3f}s)")

start_time = time.time()
model, preprocess = clip.load("ViT-B/32", device=device)
print(f"[+] Loaded model.\t\t({time.time()-start_time:.3f}s)")

with open(DATASET_DIR+'photo_ids.list', 'rb') as fp:
    photo_ids = pickle.load(fp)

photo_urls = []
with open(DATASET_DIR+"photos.tsv000") as fIn:
    reader = csv.DictReader(fIn, delimiter='\t')
    for row in reader:
        photo_urls.append([row['photo_id'], row['photo_image_url']])

id_url_dict = {}
def assign(x):
    id_url_dict[x[0]]=x[1]
ret = list(map(assign, photo_urls))


app = Flask(__name__)

@app.route('/')
def index():
	return render_template("index.html")

def search_query(inp_query, embedding_db, cpu=False, top_k=20):
	text = clip.tokenize([inp_query]).to(device)
	with torch.no_grad():
		text_features = model.encode_text(text)#.to(torch.half)
		text_features /= text_features.norm(dim=-1, keepdim=True)
		# cosine similarity as logits
		logits_per_image = (images_embeddings @ text_features.t()).squeeze()
		logits_per_image = logits_per_image.cpu().numpy()
		best_img_idx = np.argsort(logits_per_image)[::-1]
	return best_img_idx[:top_k]

cache_img_idx = {}
# TODO: Limit cache_img_idx size.
def get_resp_dicts(query):
	resp = []
	best_img_idx = cache_img_idx.get(query)
	if best_img_idx is None:
		best_img_idx = search_query(query, images_embeddings)
		cache_img_idx[query] = best_img_idx
	for idx in best_img_idx:
		im_id = photo_ids[idx]
		resp.append({"id": im_id, "url": id_url_dict[im_id]})
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

	
if __name__ == '__main__':
	app.run(debug=False)
#	app.run(host='0.0.0.0', port=31796, debug=False)