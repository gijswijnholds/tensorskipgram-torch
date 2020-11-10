from torch.utils.data import Dataset

class GS2011(Dataset):
    def __init__(self, data_fn: str, setting: str, filter: bool=False, data_sort='relatedness'):
        self.data_fn = data_fn
        self.setting = setting
        self.filter = filter
        self.data_sort = data_sort
        if os.path.exists(data_fn):
            print("Data pairs found on disk, loading...")
            self.data_pairs = load_obj_fn(data_fn)[self.setting]
            if self.filter:
                self.data_pairs = [(s1, s2, l) for (s1, s2, l) in self.data_pairs
                                   if self.has_verbs(s1) and self.has_verbs(s2)]
        else:
            print("Data pairs not found, please run create_data with a preproc.")
            self.data_pairs = None

    def __len__(self) -> int:
        return len(self.data_pairs)

    def __getitem__(self, idx: int) -> Tuple[SentenceData, SentenceData, SimilarityLabel]:
        return self.data_pairs[idx]
