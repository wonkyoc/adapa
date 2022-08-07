# JSON -> Pickle
import pickle
import json
import sys
import os

# open json file
with open(sys.argv[1], 'r') as infile:
    objs = json.loads(infile.read())

# write the pickle file
for obj in objs:
    obj["bbox"] = [float(x) for x in obj["bbox"][1:-1].split()]
    obj["score"] = float(obj["score"])
    obj["category_id"] = int(obj["category_id"])

with open(os.path.splitext(sys.argv[1])[0] + '.pkl', 'wb') as outfile:
    pickle.dump(objs, outfile)
