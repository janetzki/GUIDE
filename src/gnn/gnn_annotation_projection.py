import argparse
import sys

from my_utils.earlystopping import EarlyStopping

sys.path.insert(0, '../')
import importlib, gc
import my_utils.alignment_features as afeatures

# from my_utils.pytorch_utils import EarlyStopping

importlib.reload(afeatures)
# import gnn_utils.graph_utils as gutils

from tqdm import tqdm

import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as pyg_nn

import time
from datetime import datetime

import torch

dev = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
dev2 = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

import gensim.downloader as api
from my_utils import utils

# %%

print("Loading word2vec model")
## w2v_model = Word2Vec.load("/mounts/work/ayyoob/models/w2v/word2vec_POS_small_final_langs_10e.model")
w2v_model = api.load('glove-twitter-25')
word_vectors = torch.from_numpy(w2v_model.vectors).float()

WORDEMBEDDING = True


class Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, features, n_head=2, has_tagfreq_feature=False, ):
        super(Encoder, self).__init__()
        self.conv1 = pyg_nn.GATConv(in_channels, 2 * out_channels, heads=n_head)
        self.conv2 = pyg_nn.GATConv(2 * n_head * out_channels, out_channels, heads=n_head)
        # self.fin_lin = nn.Linear(out_channels, out_channels)

        if has_tagfreq_feature:
            if WORDEMBEDDING:
                self.feature_encoder = afeatures.FeatureEncoding(features, [normalized_tag_frequencies, word_vectors],
                                                                 dev)
            else:
                self.feature_encoder = afeatures.FeatureEncoding(features, [normalized_tag_frequencies], dev)
            # self.feature_encoder = afeatures.FeatureEncoding(features, [normalized_tag_frequencies,train_pos_labels, word_vectors])
        else:
            if WORDEMBEDDING:
                self.feature_encoder = afeatures.FeatureEncoding(features, [word_vectors], dev)
            else:
                self.feature_encoder = afeatures.FeatureEncoding(features, [], dev)
            # self.feature_encoder = afeatures.FeatureEncoding(features, [])
            # self.feature_encoder = afeatures.FeatureEncoding(features, [train_pos_labels, word_vectors])

    def forward(self, x, edge_index):
        encoded = self.feature_encoder(x, dev)  # 10x14 = #nodes x #features  ; batch_size = 1

        x = F.elu(self.conv1(encoded,
                             edge_index, ))  # 10x167 x 2x10 (#nodes x (encoded features) x #SDs x #nodes) # 242x2048 = features.sum() x hidden_size?  # 167 != 242!
        x = F.elu(self.conv2(x, edge_index))
        # return F.relu(self.fin_lin(x)), encoded
        return x, encoded


model = None
XLMR = False
POSTAG = True


def clean_memory():
    gc.collect()
    with torch.no_grad():
        torch.cuda.empty_cache()


class POSDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, n_class, skip_connection, drop_out=0):
        super(POSDecoder, self).__init__()
        self.skip_connection = skip_connection
        self.transfer = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU(), nn.Dropout(drop_out),
                                      nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Dropout(drop_out),
                                      nn.Linear(hidden_size, n_class))

    def forward(self, z, index, batch_=None):
        h = z[index, :]

        x = batch_['encoded'][index, :]
        if self.skip_connection:
            h = torch.cat((h, x), dim=1)

        res = self.transfer(h)

        return res


class POSDecoderTransformer(nn.Module):
    def __init__(self, input_size, hidden_size, n_class, residual_connection, sequence_size, drop_out=0):
        super(POSDecoderTransformer, self).__init__()
        self.sequence_size = sequence_size
        self.residual_connection = residual_connection

        # for skip connection:
        self.input_size = input_size
        n_head = int(input_size / 64)
        if n_head * 64 != input_size:
            n_head += 1
            self.input_size = n_head * 64

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.input_size, nhead=int(self.input_size / 64),
                                                        dim_feedforward=hidden_size)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=4)

        self.transfer = nn.Sequential(nn.Linear(self.input_size, self.input_size * 2), nn.ReLU(), nn.Dropout(drop_out),
                                      # TODO (original) check what happens if I remove this.
                                      nn.Linear(self.input_size * 2, n_class))

    def forward(self, z_, index, batch_):
        z = z_.to(dev2)

        x = F.pad(batch_['encoded'], (0, self.input_size - z.size(1) - batch_['encoded'].size(1))).to(dev2)

        language_based_nodes = batch_['lang_based_nodes']  # determines which node belongs to which language
        transformer_indices = batch_['transformer_indices']  # the reverse of the prev structure

        sentences = []
        for lang_nodes in language_based_nodes:  # we rearrange the nodes into sentences of each language
            if self.residual_connection:
                tensor = torch.cat((z[lang_nodes, :], x[lang_nodes, :]), dim=1)
            else:
                tensor = z[lang_nodes, :]

            try:
                tensor = F.pad(tensor, (0, 0, 0, self.sequence_size - tensor.size(0)))
            except Exception as e:
                print(self.sequence_size, tensor.size(0))
            sentences.append(tensor)

        batch = torch.stack(sentences)  # A batch contains all translations of one sentence in all training languages.
        batch = torch.transpose(batch, 0, 1)

        h = self.transformer(batch)
        h = torch.transpose(h, 0, 1)
        h = h[transformer_indices[0], transformer_indices[1],
            :]  # rearrange the nodes back to the order in which we recieved (the order that represents the graph)

        res = self.transfer(h)

        return res.to(dev)


def save_model(model, name):
    model.encoder.feature_encoder.feature_types[0] = afeatures.OneHotFeature(20, 35, 'editf')
    model.encoder.feature_encoder.feature_types[1] = afeatures.OneHotFeature(32, 256, 'position')
    model.encoder.feature_encoder.feature_types[2] = afeatures.FloatFeature(4, 'degree_centrality')
    model.encoder.feature_encoder.feature_types[3] = afeatures.FloatFeature(4, 'closeness_centrality')
    model.encoder.feature_encoder.feature_types[4] = afeatures.FloatFeature(4, 'betweenness_centrality')
    model.encoder.feature_encoder.feature_types[5] = afeatures.FloatFeature(4, 'load_centrality')
    model.encoder.feature_encoder.feature_types[6] = afeatures.FloatFeature(4, 'harmonic_centrality')
    model.encoder.feature_encoder.feature_types[7] = afeatures.OneHotFeature(32, 256, 'greedy_modularity_community')
    model.encoder.feature_encoder.feature_types[8] = afeatures.OneHotFeature(32, 256, 'community_2')
    # model.encoder.feature_encoder.feature_types[9] = afeatures.MappingFeature(100, 'word')
    # model.encoder.feature_encoder.feature_types[10] = afeatures.MappingFeature(len(postag_map), 'tag_priors', freeze=True)

    # works in DEBUG mode
    torch.save(model,
               f'../external_repos/GNN-POSTAG/dataset/small_final/gnn/checkpoint/postagging/POSTAG_{name}_' + '.pickle')


import collections

verse_accuracy_buckets = collections.defaultdict(list)
verse_accuracies = {}


def add_accuracy_to_bucket(accuracy, verse):
    bottom = int(accuracy / 5) * 5
    top = (int(accuracy / 5) + 1) * 5

    bucket = f'{bottom}-{top}'
    verse_accuracies[verse] = accuracy
    verse_accuracy_buckets[bucket].append(verse)


def test(epoch, testloader, mask_language):
    print('testing', epoch)
    model.eval()
    total = 0
    correct = 0
    total_loss = 0
    probability_sum = 0
    probability_count = 0

    data_encoder = DataEncoder(testloader, model, mask_language, dotqdm=False)

    with torch.no_grad():
        for z, verse, i, batch in data_encoder:

            target = batch['pos_classes'][0].to(dev)
            index = batch['pos_index'][0].to(dev)

            # print(target.shape, index.shape)
            preds = model.decoder(z, index, batch)
            # print(preds.shape)

            if preds.size(0) > 0:
                _, labels = torch.max(target, 1)

                actual_labels = labels != postag_map['X']
                labels = labels[actual_labels]
                preds = preds[actual_labels]

                max_probs, predicted = torch.max(torch.softmax(preds, dim=1), 1)

                loss = criterion(preds, labels)
                probability_sum += torch.sum(max_probs)
                probability_count += max_probs.size(0)
                # total_loss += loss

            verse_correct = (predicted == labels).sum().item()
            total += labels.size(0)
            correct += verse_correct
            verse_accuracy = verse_correct / (labels.size(0) + 0.00001)
            add_accuracy_to_bucket(verse_accuracy, verse)
    print(
        f'test, epoch: {epoch}, total:{total} ACC: {correct / total}, loss: {total_loss}, confidence: {probability_sum / probability_count}')
    clean_memory()
    return 1.0 - correct / total


def create_model(train_gnn_dataset, test_gnn_dataset,
                 tag_frequencies=False, use_transformers=False, train_word_embedding=False, mask_language=True,
                 residual_connection=False,
                 params=''):
    global model, criterion, optimizer, early_stopping, start_time

    if WORDEMBEDDING:
        features = train_dataset.features[:]
        features[-1].out_dim = w2v_model.vector_size
    else:
        features = train_dataset.features[:-1]

    # features.append(afeatures.PassFeature(name='posTAG', dim=len(postag_map)))
    if XLMR:
        features.append(afeatures.PassFeature(1024, 'xlmr'))
    if POSTAG:
        features.append(afeatures.PassFeature(len(postag_map), 'neighbor_tags'))
    if tag_frequencies:
        features.append(afeatures.MappingFeature(len(postag_map), 'tag_priors'))  # , freeze=True))
        # features[-2] = afeatures.MappingFeature(len(postag_map), 'tag_priors', freeze=True)
    # features[9].freeze = not train_word_embedding
    for i, feature in enumerate(features):
        print(i, vars(feature))

    train_gnn_dataset.set_transformer(use_transformers)
    test_gnn_dataset.set_transformer(use_transformers)
    train_data_loader = DataLoader(train_gnn_dataset, batch_size=1, shuffle=True)
    test_data_loader = DataLoader(test_gnn_dataset, batch_size=1, shuffle=False)

    clean_memory()
    drop_out = 0
    n_head = 1
    in_dim = sum(t.out_dim for t in features)
    print(in_dim)
    start_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    # start_time = start_time_
    delta = 0
    patience = 8
    early_stopping = EarlyStopping(verbose=True,
                                   path=f'../external_repos/GNN-POSTAG/dataset/small_final/models/gnn/checkpoint/postagging/check_point_{start_time}.pt',
                                   patience=patience, delta=delta)
    channels = 1024
    decoder_in_dim = n_head * channels + (in_dim if residual_connection else 0)

    print('len features', len(features), f'start time: {start_time}')

    if use_transformers:
        decoder = POSDecoderTransformer(decoder_in_dim, 2048, len(postag_map), residual_connection,
                                        features[1].n_classes, drop_out=drop_out).to(dev2)
    else:
        decoder = POSDecoder(decoder_in_dim, decoder_in_dim * 2, len(postag_map), residual_connection)

    model = pyg_nn.GAE(Encoder(in_dim, channels, features, n_head, has_tagfreq_feature=tag_frequencies), decoder).to(
        dev)

    if use_transformers:
        decoder.to(dev2)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00001)

    torch.set_printoptions(edgeitems=10)
    print("model params - decoder params - conv1", sum(p.numel() for p in model.parameters()),
          sum(p.numel() for p in decoder.parameters()))

    # model.load_state_dict(torch.load(f'/mounts/work/ayyoob/models/gnn/checkpoint/postagging/check_point_{start_time}.pt'))
    for epoch in range(1, 2):
        print(f"\n----------------epoch {epoch} ---------------")

        train(epoch, train_data_loader, mask_language, test_data_loader)

        if early_stopping.early_stop:
            model.load_state_dict(
                torch.load(f'/mounts/work/ayyoob/models/gnn/checkpoint/postagging/check_point_{start_time}.pt'))

        model_name = f'{len(pos_lang_list)}lngs-POSFeat{tag_frequencies}alltgts_trnsfrmr{use_transformers}6LRes{residual_connection}_trainWE{train_word_embedding}_mskLng{mask_language}_E{epoch}_{params}_{start_time}_ElyStpDlta{delta}-GA-chnls{channels}'
        model.model_name = model_name
        print('model name', model_name)
        save_model(model, model_name)
        test(epoch, test_data_loader, mask_language)
        # test_mostfreq(yor_data_loader, True, yor_gold_mostfreq_tag, yor_gold_mostfreq_index, (.shape[0], len(postag_map)))
        # test_mostfreq(tam_data_loader, True, tam_gold_mostfreq_tag, tam_gold_mostfreq_index, (w2v_model.wv.vectors.shape[0], len(postag_map)))
        # test_mostfreq(arb_data_loader, True, arb_gold_mostfreq_tag, arb_gold_mostfreq_index, (w2v_model.wv.vectors.shape[0], len(postag_map)))
        # test_mostfreq(por_data_loader, True, por_gold_mostfreq_tag, por_gold_mostfreq_index, (w2v_model.wv.vectors.shape[0], len(postag_map)))

        clean_memory()

    return model_name


def train(epoch, data_loader, mask_language, test_data_loader, max_batches=999999999):
    global optimizer
    total_loss = 0
    model.train()
    loss_multi_round = 0

    data_encoder = DataEncoder(data_loader, model, mask_language)
    optimizer.zero_grad()

    for z, verse, i, batch in data_encoder:

        target = batch['pos_classes'][0].to(dev)
        # print('labels', target)
        _, labels = torch.max(target, 1)

        index = batch['pos_index'][0].to(dev)

        preds = model.decoder(z, index, batch)

        loss = criterion(preds, labels)
        loss = loss * target.shape[0]  # TODO (original) check if this is necessary
        loss.backward()
        total_loss += loss.item()

        if (i + 1) % 5 == 0:  # Gradient accumulation
            optimizer.step()
            optimizer.zero_grad()
            # clean_memory()

        if i % 1000 == 999:
            # print(f"loss: {total_loss}")
            total_loss = 0
            val_loss = test(epoch, test_data_loader, mask_language)
            # test_mostfreq(yor_data_loader, True, yor_gold_mostfreq_tag, yor_gold_mostfreq_index, (w2v_model.wv.vectors.shape[0], len(postag_map)))
            # test_mostfreq(tam_data_loader, True, tam_gold_mostfreq_tag, tam_gold_mostfreq_index, (w2v_model.wv.vectors.shape[0], len(postag_map)))
            # test_mostfreq(arb_data_loader, True, arb_gold_mostfreq_tag, arb_gold_mostfreq_index, (w2v_model.wv.vectors.shape[0], len(postag_map)))
            # test_mostfreq(por_data_loader, True, por_gold_mostfreq_tag, por_gold_mostfreq_index, (w2v_model.wv.vectors.shape[0], len(postag_map)))
            print(
                '----------------------------------------------------------------------------------------------------------')
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            model.train()
            clean_memory()

        if i == max_batches:
            break

    print(f"total train loss: {total_loss}")


from torch.utils.data import Dataset, DataLoader
import random

utils.graph_dataset_path = '../external_repos/GNN-POSTAG/dataset/small_final/'
train_dataset = torch.load(utils.graph_dataset_path + f'dataset_forpos1_word.torch.bin')

new_testament_verses = []
old_testament_verses = []
testaments = 'all'

starts = set()
for verse in train_dataset.accepted_verses:
    starts.add(verse[0])
    if verse[0] in ['4', '5', '6']:
        new_testament_verses.append(verse)
    else:
        old_testament_verses.append(verse)


class POSTAGGNNDataset(Dataset):
    def __init__(self, dataset, verses, edit_files, alignments, node_cover, pos_labels, data_dir, create_data=False,
                 group_size=20, transformer=True):
        self.node_cover = node_cover
        self.pos_labels = pos_labels
        self.data_dir = data_dir
        self.items = self.calculate_size(verses, group_size, node_cover)
        self.dataset = dataset
        self.editions = edit_files

        if create_data:
            self.calculate_verse_stats(verses, edit_files, alignments, dataset, data_dir)

        # self.pool = Pool(4)
        self.transformer = transformer

    def set_transformer(self, transformer):
        self.transformer = transformer

    def calculate_size(self, verses, group_size, node_cover):
        res = []
        self.res_new_testament = []
        self.res_old_testament = []
        for verse in verses:
            covered_nodes = node_cover[verse]
            random.shuffle(covered_nodes)
            items = [covered_nodes[i:i + group_size] for i in range(0, len(covered_nodes), group_size)]
            res.extend([(verse, i) for i in items])

            if verse in new_testament_verses:
                self.res_new_testament.extend([(verse, i) for i in items])
            if verse in old_testament_verses:
                self.res_old_testament.extend([(verse, i) for i in items])

        return res

    def __len__(self):
        global testaments
        if testaments == 'new':
            return len(self.res_new_testament)
        elif testaments == 'old':
            return len(self.res_old_testament)
        return len(self.items)

    def __getitem__(self, idx):
        global testaments
        start_time = time.time()
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if testaments == 'new':
            verse, nodes = self.res_new_testament[idx]
        elif testaments == 'old':
            verse, nodes = self.res_old_testament[idx]
        else:
            verse, nodes = self.items[idx]

        self.verse_info = {verse: torch.load(f'{self.data_dir}/verses/{verse}_info.torch.bin')}

        padding = self.verse_info[verse]['padding']

        if self.transformer:
            language_based_nodes, transformer_indices = posutil.get_language_based_nodes(self.dataset.nodes_map, verse,
                                                                                         nodes, padding)
        else:
            language_based_nodes, transformer_indices = 0, 0

        if XLMR:
            xlmr_emb = get_xlmr_embeddings(self.dataset.nodes_map, verse, self.editions, w2v_model,
                                           self.verse_info[verse]['x'].size(0), xlmr, self.data_dir,
                                           self.verse_info[verse]['x'][:, 9].long())
            self.verse_info[verse]['x'] = torch.cat((self.verse_info[verse]['x'], xlmr_emb), dim=1)
        # # Add POSTAG to set of features
        if POSTAG:
            postags = self.pos_labels[verse][padding: self.verse_info[verse]['x'].size(0) + padding, :]
            postags = postags.detach().clone()
            postags[torch.LongTensor(nodes) - padding, :] = 0
            self.verse_info[verse]['x'] = torch.cat((self.verse_info[verse]['x'], postags), dim=1)

        # Add token id as a feature, used to extract token information (like token's tag distribution)
        word_number = self.verse_info[verse]['x'][:, 9]
        word_number = torch.unsqueeze(word_number, 1)
        self.verse_info[verse]['x'] = torch.cat((self.verse_info[verse]['x'], word_number), dim=1)
        edge_index = self.verse_info[verse]['edge_index']
        # print('dataset time:', time.time()-start_time)

        if not WORDEMBEDDING:
            self.verse_info[verse]['x'] = torch.cat(
                (self.verse_info[verse]['x'][:, :9], self.verse_info[verse]['x'][:, 10:]), dim=1)
        return {'verse': verse, 'x': self.verse_info[verse]['x'],
                'edge_index': edge_index if torch.is_tensor(edge_index) else torch.tensor(edge_index, dtype=torch.long),
                'pos_classes': self.pos_labels[verse][nodes, :], 'pos_index': torch.LongTensor(nodes) - padding,
                'padding': padding, 'lang_based_nodes': language_based_nodes,
                'transformer_indices': transformer_indices}


langs = ['eng', 'fra']
editions = {'eng': 'bid-eng-dby', 'fra': 'bid-fra-fob'}

current_editions = []
for lang in langs:
    if editions[lang] not in current_editions:
        current_editions.append(editions[lang])


def create_me_a_gnn_dataset(node_covers, labels, group_size=100, editions=current_editions,
                            verses=train_dataset.accepted_verses):
    train_ds = POSTAGGNNDataset(train_dataset, verses, editions, {}, node_covers[0], labels[0],
                                utils.graph_dataset_path + '/', group_size=group_size)

    return train_ds


import collections


def create_structures(dataset, all_tags):
    pos_labels = {}
    pos_node_cover = collections.defaultdict(list)

    for lang in all_tags:
        for sent_id in all_tags[lang]:
            sent_tags = all_tags[lang][sent_id]
            if sent_id not in pos_labels:
                pos_labels[sent_id] = torch.zeros(dataset.verse_lengthes[sent_id], len(postag_map))
            for w_i in range(len(sent_tags)):
                if w_i not in dataset.nodes_map[lang][sent_id]:
                    continue
                pos_labels[sent_id][dataset.nodes_map[lang][sent_id][w_i], sent_tags[w_i]] = 1
                pos_node_cover[sent_id].append(dataset.nodes_map[lang][sent_id][w_i])
    return pos_labels, pos_node_cover


import codecs

postag_map = {'Moon': 1, 'Star': 2, 'X': 0, 'None': 0}
criterion = nn.CrossEntropyLoss(ignore_index=postag_map['X'])


def get_pos_tags(dataset, pos_lang_list):
    all_tags = {}
    for lang in pos_lang_list:
        if lang not in dataset.nodes_map:
            continue
        all_tags[lang] = {}

        base_path = utils.graph_dataset_path

        # if os.path.exists(F"/mounts/work/silvia/POS/TAGGED_LANGS/{lang}.conllu"):
        # 	base_path = F"/mounts/work/silvia/POS/TAGGED_LANGS/"
        # else:
        # 	base_path = F"/mounts/work/mjalili/projects/gnn-align/data/pbc_pos_tags/"

        with codecs.open(F"{base_path}/tokenized_bibles/{lang}_sd.conllu", "r", "utf-8") as lang_pos:
            tag_sent = []
            sent_id = ""
            for sline in lang_pos:
                sline = sline.strip()
                if sline == "":
                    if sent_id not in dataset.nodes_map[lang]:
                        tag_sent = []
                        sent_id = ""
                        continue

                    all_tags[lang][sent_id] = [postag_map[p[3]] for p in tag_sent]
                    tag_sent = []
                    sent_id = ""
                elif "# verse_id" in sline or '# sent_id' in sline:
                    sent_id = sline.split()[-1]
                elif sline[0] == "#":
                    continue
                else:
                    tag_sent.append(sline.split("\t"))

    pos_labels, pos_node_cover = create_structures(dataset, all_tags)
    return pos_labels, pos_node_cover


def get_db_nodecount(dataset):
    res = 0
    for lang in dataset.nodes_map.values():
        for verse in lang.values():
            res += len(verse)

    return res


def get_language_nodes(dataset, lang_list, sentences):
    node_count = get_db_nodecount(dataset)
    pos_labels = {}

    pos_node_cover = collections.defaultdict(list)
    for lang in lang_list:
        if lang in dataset.nodes_map:
            for sentence in sentences:
                if sentence not in pos_labels:
                    pos_labels[sentence] = torch.zeros(dataset.verse_lengthes[sentence], len(postag_map))
                if sentence in dataset.nodes_map[lang]:
                    for tok in dataset.nodes_map[lang][sentence]:
                        pos_node_cover[sentence].append(dataset.nodes_map[lang][sentence][tok])

    return pos_labels, pos_node_cover


def get_data_loadrs_for_target_editions(target_editions, dataset, verses, data_dir, transformer):
    target_pos_labels, target_pos_node_cover = get_language_nodes(dataset, target_editions, verses)
    gnn_dataset_target_pos = POSTAGGNNDataset(dataset, verses, None, {}, target_pos_node_cover, target_pos_labels,
                                              data_dir, group_size=50000, transformer=transformer)
    target_data_loader = DataLoader(gnn_dataset_target_pos, batch_size=1, shuffle=False)

    return target_data_loader


class DataEncoder():

    def __init__(self, data_loader, model, mask_language, dotqdm=True):
        self.data_loader = data_loader
        self.model = model
        self.mask_language = mask_language
        self.dotqdm = dotqdm

    def __iter__(self):

        for i, batch in enumerate(tqdm(self.data_loader)) if self.dotqdm else enumerate(self.data_loader):
            x = batch['x'][0].to(dev)  # initial features (not encoded)
            edge_index = batch['edge_index'][0].to(dev)
            # print(edge_index.shape)
            verse = batch['verse'][0]

            # index = batch['pos_index'][0].to(dev)
            # print('x', x[index, 10:28])
            # if verse in masked_verses:
            #     continue

            try:
                if self.mask_language:
                    x[:, 0] = 0
                z, encoded = self.model.encode(x, edge_index)  # Z will be the output of the GNN
                batch['encoded'] = encoded
            except Exception as e:
                global sag, khar, gav
                sag, khar, gav = (i, batch, verse)
                print(sag, khar, gav)
                print(e)
                1 / 0

            yield z, verse, i, batch


import postag_utils as posutil

shuffled_verses = train_dataset.accepted_verses[:]
random.shuffle(shuffled_verses)

pos_lang_list = ['bid-eng-DBY-1000', 'bid-fra-fob-1000']  # ['eng-x-bible-web', 'fra-x-bible-fob']
pos_val_lang_list = ['bid-eng-DBY-1000', 'bid-fra-fob-1000']
pos_test_lang_list = ['bid-eng-DBY-1000', 'bid-fra-fob-1000']
train_pos_labels, train_pos_node_cover = get_pos_tags(train_dataset, pos_lang_list)
val_pos_labels, val_pos_node_cover = get_pos_tags(train_dataset, pos_val_lang_list)

target_editions = []
for edition in current_editions:
    if edition not in pos_lang_list:  # and (edition in pos_val_lang_list or edition in pos_test_lang_list):
        target_editions.append(edition)

target_data_loader_train = get_data_loadrs_for_target_editions(target_editions, train_dataset,
                                                               train_dataset.accepted_verses,
                                                               utils.graph_dataset_path + '/',
                                                               transformer=False)

gnn_dataset_train_pos_bigbatch = create_me_a_gnn_dataset([train_pos_node_cover], [train_pos_labels],
                                                         group_size=2, verses=shuffled_verses[:])
gnn_dataset_val_pos = create_me_a_gnn_dataset([val_pos_node_cover], [val_pos_labels], group_size=2,
                                              verses=shuffled_verses[:3])
gnn_dataset_test_pos = create_me_a_gnn_dataset([val_pos_node_cover], [val_pos_labels], group_size=2,
                                               verses=train_dataset.accepted_verses)

train_data_loader_bigbatch = DataLoader(gnn_dataset_train_pos_bigbatch, batch_size=1, shuffle=False)
val_data_loader_pos = DataLoader(gnn_dataset_val_pos, batch_size=1, shuffle=False)
test_data_loader_pos = DataLoader(gnn_dataset_test_pos, batch_size=1, shuffle=False)
threshold = 0.95
type_check = False

##! Uncomment when data has been updated
# tag_frequencies_source = torch.zeros(w2v_model.vectors.shape[0], len(postag_map)) # tag_frequencies_source = tag_frequencies - tag_frequencies_target
# res_ = posutil.get_tag_frequencies_node_tags(model, [train_pos_node_cover], [train_pos_labels], len(postag_map),
#                                              w2v_model.vectors.shape[0],
#                                              [target_data_loader_train], [train_data_loader_bigbatch], DataEncoder,
#                                              source_tag_frequencies=tag_frequencies_source,
#                                              target_train_treshold=threshold, type_check=type_check)
#
# print('saving at',
#       f'../external_repos/GNN-POSTAG/dataset/small_final/tag_frequencies_th{threshold}_typchk{type_check}.torch.bin')
# torch.save(res_,
#            f'../external_repos/GNN-POSTAG/dataset/small_final/tag_frequencies_th{threshold}_typchk{type_check}.torch.bin')
# exit()

res_ = torch.load(
    f'../external_repos/GNN-POSTAG/dataset/small_final/tag_frequencies_th{threshold}_typchk{type_check}.torch.bin')
tag_frequencies, tag_frequencies_target, pos_node_cover_exts, pos_label_exts, tag_based_stats = res_
train_pos_node_covers_ext, train_pos_labels_ext = pos_node_cover_exts[0], pos_label_exts[0]

tag_frequencies_source = tag_frequencies - tag_frequencies_target
word_frequencies_target = torch.sum(tag_frequencies_target.to(torch.device('cpu')), dim=1)
tag_frequencies = tag_frequencies_source + tag_frequencies_target
tag_frequencies_copy = tag_frequencies.detach().clone()

tag_frequencies_copy[torch.logical_and(word_frequencies_target > 0.1, word_frequencies_target < 3), :] = 0.0000001

# We have to give uniform noise to some training examples to prevent the model from returning one of the most frequent tags always!!
uniform_noise = torch.BoolTensor(tag_frequencies.size(0))
uniform_noise[:] = True
shuffle_tensor = torch.randperm(tag_frequencies.size(0))[:int(tag_frequencies.size(0) * 0.7)]
uniform_noise[shuffle_tensor] = False
tag_frequencies_copy[torch.logical_and(uniform_noise, word_frequencies_target < 0.1), :] = 0.0000001
sm = torch.sum(tag_frequencies_copy, dim=1)
normalized_tag_frequencies = (tag_frequencies_copy.transpose(1, 0) / sm).transpose(1, 0)

gnn_dataset_val_pos = create_me_a_gnn_dataset([val_pos_node_cover], [val_pos_labels], group_size=10000,
                                              verses=shuffled_verses[:1000])

mask_language = True
params = argparse.Namespace()

gnn_dataset_train_pos_ext = create_me_a_gnn_dataset([train_pos_node_covers_ext], [train_pos_labels_ext],
                                                    group_size=16)
model_name = create_model(gnn_dataset_train_pos_ext, gnn_dataset_val_pos,
                          train_word_embedding=False, mask_language=mask_language, use_transformers=True,
                          tag_frequencies=True, params=f'traintgt{threshold}_Typchck{type_check}_small_NOXLMR',
                          residual_connection=False)

# %%
m_path = 'POSTAG_2lngs-POSFeatTruealltgts_trnsfrmrTrue6LResFalse_trainWEFalse_mskLngTrue_E1_Namespace()_20230512-113529_ElyStpDlta0-GA-chnls1024_.pickle'
model = torch.load(f'../external_repos/GNN-POSTAG/dataset/small_final/gnn/checkpoint/postagging/{m_path}',
                   map_location=torch.device('cpu'))
model.to(dev)
model.decoder.to(dev2)
gnn_dataset_val_pos.set_transformer(True)
test(0, val_data_loader_pos, True)

gnn_dataset_test_pos.set_transformer(False)
gen_langs = pos_test_lang_list[:]
gen_langs.extend(pos_val_lang_list)
for lang in gen_langs:
    posutil.generate_target_lang_tags_onedataset(model, lang, model.model_name, mask_language,
                                                 train_dataset, test_data_loader_pos, DataEncoder, postag_map['X'])

model = None
decoder = None
clean_memory()
