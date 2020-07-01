import numpy as np
import sys
import os
import torch
from flask import Flask, request, jsonify
import json

from p2ch13.model_cls import LunaModel

app = Flask(__name__)

model = LunaModel()
model.load_state_dict(torch.load(sys.argv[1],
                                 map_location='cpu')['model_state'])
model.eval()

def run_inference(in_tensor):
    with torch.no_grad():
        # LunaModel takes a batch and outputs a tuple (scores, probs)
        out_tensor = model(in_tensor.unsqueeze(0))[1].squeeze(0)
    probs = out_tensor.tolist()
    out = {'prob_malignant': probs[1]}
    return out

@app.route("/predict", methods=["POST"])
def predict():
    meta = json.load(request.files['meta'])
    blob = request.files['blob'].read()
    in_tensor = torch.from_numpy(np.frombuffer(
        blob, dtype=np.float32))
    in_tensor = in_tensor.view(*meta['shape'])
    out = run_inference(in_tensor)
    return jsonify(out)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
    print (sys.argv[1])
