from tensorskipgram.config import preproc_fn, svo_triples_fn, verblist_fn, noun_space_fn
from tensorskipgram.config import model_path_subj, model_path_obj


def main_old(bs=1, lr=0.005, ne=5):
    my_preproc = Preprocessor(preproc_fn, noun_space_fn, svo_triples_fn, verblist_fn)
    subj_i2w = my_preproc.preproc['subj']['i2w']
    obj_i2w = my_preproc.preproc['obj']['i2w']
    verb_i2v = my_preproc.preproc['verb']['i2v']
    lower2upper = my_preproc.preproc['l2u']

    # noun_vocab_size = len(subj_i2w)
    # context_vocab_size = len(obj_i2w)
    noun_vocab_size = len(obj_i2w)
    context_vocab_size = len(subj_i2w)
    functor_vocab_size = len(verb_i2v)
    noun_matrix = create_noun_matrix(noun_space_fn, obj_i2w, lower2upper)
    # noun_matrix = create_noun_matrix(subj_i2w, lower2upper)
    noun_matrix = torch.tensor(noun_matrix, dtype=torch.float32)


    # print("Preparing data loader...")
    # subj_dataset = MatrixSkipgramDataset(subj_data_fn, arg='subject', negk=5)
    # subj_dataloader6 = DataLoader(subj_dataset, shuffle=True, batch_size=1)
    print("Preparing data loader...")
    subj_dataset = MatrixSkipgramDataset(subj_data_fn, arg='subject', negk=5)
    subj_dataloader = DataLoader(subj_dataset, shuffle=True, batch_size=bs)
    # print("Preparing data loader...")
    # obj_dataset = MatrixSkipgramDataset(obj_data_fn, arg='object', negk=5)
    # obj_dataloader = DataLoader(obj_dataset, shuffle=True, batch_size=bs)

    print("Training model...")

    subj_matskipgram_modelGPU = MatrixSkipgram(noun_vocab_size=noun_vocab_size,
                                              functor_vocab_size=functor_vocab_size,
                                              context_vocab_size=context_vocab_size,
                                              embed_size=100, noun_matrix=noun_matrix)

    # obj_matskipgram_modelGPU.to('cuda')
    subj_matskipgram_modelGPU.to(1)
    optGPU = torch.optim.Adam(subj_matskipgram_modelGPU.parameters(), lr=lr)
    loss_fnGPU = torch.nn.BCEWithLogitsLoss()

    NUM_EPOCHS = ne
    epoch_losses = []
    for i in range(NUM_EPOCHS):
        epoch_loss = train_epoch(subj_matskipgram_modelGPU, subj_dataloader,
                                 loss_fnGPU, optGPU, device=1, epoch_idx=i+1)
                                 # loss_fnGPU, optGPU, device='cuda', epoch_idx=i+1)
        # epoch_loss = train_epoch(subj_matskipgram_modelGPU, obj_dataloader6,
                                 # loss_fnGPU, optGPU, device='cuda', epoch_idx=i+1)
        epoch_losses.append(epoch_loss)
        print("Saving model...")
        e = i+1
        torch.save(subj_matskipgram_modelGPU, model_path_subj+f'_bs={bs}_lr={lr}_epoch{e}.p')
        print("Done saving model, ready for another epoch!")

    return epoch_losses

def loadArgAnalysers():
    subj_preprocFileName = '/import/gijs-shared/gijs/skipprob_data/preproc_sick_subj.pkl'
    obj_preprocFileName = '/import/gijs-shared/gijs/skipprob_data/preproc_sick_obj.pkl'
    print("Loading preprocs...")
    subj_preproc = load_obj_fn(subj_preprocFileName)
    obj_preproc = load_obj_fn(obj_preprocFileName)

    subj_i2w, subj_w2i, subj_i2c, subj_i2ns = (subj_preproc['index2word'],
                                               subj_preproc['word2index'],
                                               subj_preproc['index2count'],
                                               subj_preproc['index2negsample'])

    obj_i2w, obj_w2i, obj_i2c, obj_i2ns = (obj_preproc['index2word'],
                                           obj_preproc['word2index'],
                                           obj_preproc['index2count'],
                                           obj_preproc['index2negsample'])
    return (subj_i2w, subj_w2i, subj_i2c, subj_i2ns,
            obj_i2w, obj_w2i, obj_i2c, obj_i2ns)
            
# subjDataset = MatrixSkipgramDataset(sick_subj_data_fn_subj, arg='subject', negk=5)
# objDataset = MatrixSkipgramDataset(sick_obj_data_fn_obj, arg='object')

# subj_dataloader6 = DataLoader(subjDataset,
#                               shuffle=True,
#                               batch_size=1)
#
# subj_dataloader48 = DataLoader(subjDataset,
#                                shuffle=True,
#                                batch_size=8)
#
# subj_dataloader96 = DataLoader(subjDataset,
#                                shuffle=True,
#                                batch_size=16)
#
# subj_dataloader96 = DataLoader(subjDataset,
#                                shuffle=True,
#                                batch_size=64)
# print("Loading analysers...")
# (subj_i2w, subj_w2i, subj_i2c, subj_i2ns,
#  obj_i2w, obj_w2i, obj_i2c, obj_i2ns) = \
#     loadArgAnalysers()

# noun_vocab_size = len(obj_i2w)
# context_vocab_size = len(subj_i2w)
# nounMatrix = createNounMatrix(obj_i2w)
# nounMatrix = torch.tensor(nounMatrix, dtype=torch.float32)

def main():
  pass
