import torch
from tensorskipgram.preprocessing import Preprocessor
from tensorskipgram.config \
    import preproc_fn, svo_triples_fn, verblist_fn, noun_space_fn
from tensorskipgram.config \
    import model_path_subj, model_path_obj, subj_data_fn, obj_data_fn
from tensorskipgram.training.model import MatrixSkipgram
from tensorskipgram.training.dataset import MatrixSkipgramDataset
from tensorskipgram.training.trainer import train_epoch
from torch.utils.data import DataLoader


def prepare_model(arg: str, context: str, space_fn: str):
    """Prepare a matrix-skipgram model with a preprocessor and a noun space."""
    assert arg in ['subj', 'obj']
    assert context in ['subj', 'obj']
    assert arg != context
    preprocessor = Preprocessor(preproc_fn, space_fn, svo_triples_fn,
                                verblist_fn).preproc
    arg_i2w, context_i2w = preprocessor[arg]['i2w'], preprocessor[context]['i2w']
    verb_i2v, lower2upper = preprocessor['verb']['i2v'], preprocessor['l2u']
    noun_vocab_size, context_vocab_size = len(arg_i2w), len(context_i2w)
    functor_vocab_size = len(verb_i2v)
    noun_matrix_np = create_noun_matrix(space_fn, obj_i2w, lower2upper)
    noun_matrix = torch.tensor(noun_matrix_np, dtype=torch.float32)
    return MatrixSkipgram(noun_vocab_size=noun_vocab_size,
                          functor_vocab_size=functor_vocab_size,
                          context_vocab_size=context_vocab_size,
                          embed_size=100, noun_matrix=noun_matrix)


def train_model(arg: str, context: str, space_fn: str, bs: int, lr: float):
    print("Preparing model...")
    model = prepare_model(arg, context, space_fn)

    print("Preparing data loader...")
    if arg == 'subj':
        dataset = MatrixSkipgramDataset(subj_data_fn, arg='subject', negk=5)
        dataloader = DataLoader(dataset, shuffle=True, batch_size=bs)
    elif arg == 'obj':
        dataset = MatrixSkipgramDataset(obj_data_fn, arg='object', negk=5)
        dataloader = DataLoader(dataset, shuffle=True, batch_size=bs)

    model.to(1)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    NUM_EPOCHS = 1
    epoch_losses = []
    for i in range(NUM_EPOCHS):
        epoch_loss = train_epoch(model, dataloader,
                                 loss_fn, optimiser, device=1, epoch_idx=i+1)
        epoch_losses.append(epoch_loss)
        print("Saving model...")
        e = i+1
        torch.save(model, model_path_subj+f'_bs={bs}_lr={lr}_epoch{e}.p')
        print("Done saving model, ready for another epoch!")

    return epoch_losses


def main():
    train_model('subj', 'obj', noun_space_fn, 11, 0.001)
    train_model('obj', 'subj', noun_space_fn, 11, 0.001)
