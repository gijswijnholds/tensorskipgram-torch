import torch
from typing import List
from tensorskipgram.preprocessing.training_data_creator import Preprocessor
from tensorskipgram.config \
    import preproc_fn, svo_triples_fn, verblist_fn, noun_space_fn
from tensorskipgram.config \
    import model_path_subj, model_path_obj, subj_data_fn, obj_data_fn
from tensorskipgram.training.model import MatrixSkipgram
from tensorskipgram.training.dataset \
    import MatrixSkipgramDataset, create_noun_matrix
from tensorskipgram.training.trainer import train_epoch
from torch.utils.data import DataLoader


def prepare_model(arg: str, context: str, space_fn: str,
                  preproc_filename: str, triples_fn: str, verbs_fn: str):
    """Prepare a matrix-skipgram model with a preprocessor and a noun space."""
    assert arg in ['subj', 'obj']
    assert context in ['subj', 'obj']
    assert arg != context
    preprocessor = Preprocessor(preproc_filename, space_fn, triples_fn, verbs_fn).preproc
    arg_i2w, context_i2w = preprocessor[arg]['i2w'], preprocessor[context]['i2w']
    verb_i2v, lower2upper = preprocessor['verb']['i2v'], preprocessor['l2u']
    noun_vocab_size, context_vocab_size = len(arg_i2w), len(context_i2w)
    functor_vocab_size = len(verb_i2v)
    noun_matrix_np = create_noun_matrix(space_fn, arg_i2w, lower2upper)
    noun_matrix = torch.tensor(noun_matrix_np, dtype=torch.float32)
    return MatrixSkipgram(noun_vocab_size=noun_vocab_size,
                          functor_vocab_size=functor_vocab_size,
                          context_vocab_size=context_vocab_size,
                          embed_size=100, noun_matrix=noun_matrix)


def train_model(space_fn: str, model_path: str, arg_data_fn: str, arg: str, context: str,
                neg_k: int, batch_size: int, learning_rate: float, epochs: int,
                device: str, preproc_filename: str, triples_fn: str, verbs_fn: str) -> List[float]:
    """Train a matrix skipgram model."""
    print("Preparing model...")
    model = prepare_model(arg, context, space_fn, preproc_filename, triples_fn, verbs_fn)

    print("Preparing data loader...")
    dataset = MatrixSkipgramDataset(arg_data_fn, arg=context, negk=neg_k)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size)

    print("Preparing optimiser + loss function...")
    model.to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    print("Perform training...")
    epoch_losses = []
    for i in range(epochs):
        epoch_loss = train_epoch(model, dataloader, loss_fn, optimiser,
                                 device=device, epoch_idx=i+1)
        epoch_losses.append(epoch_loss)
        print("Saving model...")
        e = i+1
        model_save_path = model_path + f'_bs={batch_size}_lr={learning_rate}_epoch{e}.p'
        torch.save(model, model_save_path)
        print("Done saving model, ready for another epoch!")

    return epoch_losses


def main():
    train_model(noun_space_fn, model_path_subj, obj_data_fn, arg='subj', context='obj',
                neg_k=5, batch_size=11, learning_rate=0.001, epochs=1, device='cuda',
                preproc_filename=preproc_fn, triples_fn=svo_triples_fn,
                verbs_fn=verblist_fn)
    train_model(noun_space_fn, model_path_obj, subj_data_fn, arg='obj', context='subj',
                neg_k=5, batch_size=11, learning_rate=0.001, epochs=1, device='cuda',
                preproc_filename=preproc_fn, triples_fn=svo_triples_fn,
                verbs_fn=verblist_fn)


# fn1 = 'skipprob_data/training_data_combined_subject/train_data_proper_asym_ns=5.npy'
# fn2 = verb_data/subj_train_data_1160.p
