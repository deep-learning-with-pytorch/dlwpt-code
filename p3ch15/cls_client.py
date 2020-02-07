import requests
import json
import io
import torch

im, cl, id, pos = torch.load('data/p3ch15/cls_val_example.pt')

meta = io.StringIO(json.dumps({'shape': list(im.shape)}))
data = io.BytesIO(bytearray(im.numpy()))
r = requests.post('http://localhost:8000/predict',
                  files={'meta': meta, 'blob' : data})
response = json.loads(r.content)

print("Model predicted probability of being maignant:", response['prob_malignant'])
