import h5py

class ANCSHEvaluator:
    def __init__(self, cfg, combined_results_path):
        self.cfg = cfg
        self.f_combined = h5py.File(combined_results_path, "r+")
        self.instances = sorted(self.f_combined.keys())
        self.results = {}

    def process(self):
        pass


    def print_and_save(self):
        pass