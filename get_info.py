import pickle
from os.path import isfile


category = [
        "person",       # 0
        "bicycle",      # 1
        "car",          # 2
        "motorcycle",   # 3
        "bus",          # 5
        "truck",        # 7
        "traffic_light",# 9
        "stop_sign",    # 11
        ]

def replace_id(id):
    return id + 1802


def main():
    #bbox_path = "test-1d676737.pkl"
    bbox_path = "results_ccf_marked.pkl"
    
    # load bbox
    with open(bbox_path, "rb") as f:
        data = pickle.load(f)

    #raw_objs = []
    #for d in data:
    #    if d["image_id"] > 2270:
    #        break
    #    if d["image_id"] >= 1802:
    #        raw_objs.append(d)

    #out_path = "/home/wonkyoc/git/system/test-1d676737.pkl"
    #if not isfile(out_path):
    #    pickle.dump(raw_objs, open(out_path, "wb"))
    print(data[0])
    1/0
    
    image_id = replace_id(0)

    for d in data:
        if image_id == d["image_id"]:
            obj_type = category[d["category_id"]]
            print(obj_type, d)



if __name__ == "__main__":
    main()
