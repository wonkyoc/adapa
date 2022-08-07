import pickle
import itertools


class Frame:
    def __init__(self, frame_id, obj):
        self.frame_id = frame_id
        self.objects = [obj]


class Object:
    categories = [
            "person",       # 0
            "bicycle",      # 1
            "car",          # 2
            "motorcycle",   # 3
            "bus",          # 5
            "truck",        # 7
            "traffic_light",    # 9
            "stop_sign",    # 11
            ]
    newid = itertools.count()

    def __init__(self, image_id, bbox, score, category_id, velocity, uncertainty, 
                 marked):
        self.object_id = next(Object.newid)
        self.image_id = image_id
        self.bbox = bbox
        self.score = score
        self.category_id = category_id
        self.category = self.categories[category_id]
        self.velocity = velocity
        self.uncertainty = uncertainty
        self.marked = marked

        self.left = False   # is this left out from a frame?


class Detector:
    undefined = -1

    def __init__(self, name, scale):
        self.name = name
        self.scale = scale
        self.latency = self.undefined


class ADS:
    #bbox_path = "data/mrcnn50_nm_s0.5/val/results_ccf_marked.pkl"
    bbox_path = "results_ccf_marked.pkl"
    detector = None
    threshold = 0.05
    image_size = 1920 * 1200
    #ego_bbox = [800, 600, 500, 150]   # estimated location

    def __init__(self):
        self.frames = []
        self.detected_objects = []   # a list of detected objects. only for programming abst.

    def lookup_frame(self, frame_id):
        frame = None

        for f in self.frames:
            # return if frame exists
            if f.frame_id == frame_id:
                frame = f
                return frame

        return frame

    def read_bbox(self):
        # load bbox
        with open(self.bbox_path, "rb") as f:
            data = pickle.load(f)

        # extract only from '1d676737-4110-3f7e-bec0-0c90f74c248f'
        # our target frames: 1802 <= image_id <= 2270
        raw_objs = []
        for d in data:
            if d["image_id"] > 2270:
                break
            if d["image_id"] >= 1802:
                raw_objs.append(d)

        for obj in raw_objs:
            image_id = obj["image_id"]

            frame = self.lookup_frame(image_id)
            if frame is None:
                frame = Frame(image_id, obj)
                self.frames.append(frame)
            else:
                frame.objects.append(obj)

    def perception(self):
        runtime = 0
        for frame in self.frames:
            # we dont have any objs so full-process
            if len(self.detected_objects) < 1:
                for obj in frame.objects:
                    detected = Object(
                            obj["image_id"],
                            obj["bbox"],
                            obj["score"],
                            obj["category_id"],
                            obj["velocity"],
                            obj["uncertainty"],
                            obj["marked"],
                            )
                    self.detected_objects.append(detected)
                # add full latency
                runtime += self.detector.latency
                continue

            for old in self.detected_objects:
                # assume all objs will be deleted
                old.left = True

            # add a new obj or update the existed
            for obj in frame.objects:
                new_x = obj["bbox"][0]
                new_y = obj["bbox"][1]
                new_category_id = obj["category_id"]
                isNew = True
                for old in self.detected_objects:
                    if old.category_id != new_category_id:
                        continue

                    # if category is the same, check (x,y) range
                    old_x = old.bbox[0]
                    old_y = old.bbox[1]

                    upper_t = 1.0 + self.threshold
                    lower_t = 1.0 - self.threshold
                
                    if (new_x <= old_x * upper_t) and (new_x >= old_x * lower_t):
                        if (new_y <= old_y * upper_t) and (new_y >= old_y * lower_t):
                            # object is possibly the same then update bbox
                            old.bbox = obj["bbox"]
                            old.score = obj["score"]
                            old.velocity = obj["velocity"]
                            old.uncertainty = obj["uncertainty"]
                            old.marked = obj["marked"]
                            old.left = False
                            isNew = False
                            break
                # add a new obj in the list
                if isNew == True:
                    detected = Object(
                            obj["image_id"],
                            obj["bbox"],
                            obj["score"],
                            obj["category_id"],
                            obj["velocity"],
                            obj["uncertainty"],
                            obj["marked"],
                            )
                    self.detected_objects.append(detected)

                    # add latency for a new object
                    area = detected.bbox[2] * detected.bbox[3]
                    runtime += self.detector.latency * (area / self.image_size)

            # remove undetected obj in the current frame
            self.detected_objects = [x for x in self.detected_objects if x.left == False]

            # estimate processing time objects
            # this only considers important objects
            for obj in self.detected_objects:
                if obj.marked == 1:
                    area = obj.bbox[2] * obj.bbox[3]
                    runtime += self.detector.latency * (area / self.image_size)
                # else:
                #     # rest of objects...
                #      area = obj.bbox[2] * obj.bbox[3]
                #      runtime += (self.detector.latency * (area / self.image_size))

        # -- End of processing frames -- #
        print("total_frames (n):        %d" % (len(self.frames)))
        print("total_runtime (ms):       %.2f" % (runtime))
        print("per_frame_latency (ms):   %.2f" % (runtime / len(self.frames)))
            
    # start system in pipeline manner
    def run(self):
        self.perception()


def main():
    # initialization
    ads = ADS()

    # after read_bbox() frames will load objects on the memory but they are not
    # for programming abstraction yet
    ads.read_bbox()
    detector = Detector("efficientDet", "d0")

    # latency was profiled
    detector.latency = 31.4
    ads.detector = detector

    # simulated pipeline
    ads.run()


if __name__ == "__main__":
    main()
