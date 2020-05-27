import pickle


def dump_obj_fn(obj, fn):
    with open(fn, 'wb') as out_file:
        pickle.dump(obj, out_file)


def load_obj_fn(obj, fn):
    with open(fn, 'rb') as in_file:
        data = pickle.load(in_file)
    return data
