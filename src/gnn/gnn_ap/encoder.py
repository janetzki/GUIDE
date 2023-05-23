import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn

import my_utils.alignment_features as afeatures

WORDEMBEDDING = False

dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dev2 = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


class Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, features, n_head=2, has_tagfreq_feature=False,
                 normalized_tag_frequencies=None, word_vectors=None):
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
        encoded = self.feature_encoder(x,
                                       dev)  # 10 x 14 = #nodes (in one verse and all langs) x #features ; batch_size = 1
        # 47 x 7419 = #nodes (in one verse and all langs) x #QIDs?!
        # 28 x 15 = #nodes (in one verse and all langs) x ??

        x = F.elu(self.conv1(encoded,
                             edge_index, ))  # 10x167 x 2x10 (#nodes x (encoded features) x #SDs x #nodes) # 242x2048 = features.sum() x hidden_size?  # 167 != 242!
        x = F.elu(self.conv2(x, edge_index))
        # return F.relu(self.fin_lin(x)), encoded
        return x, encoded


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
                                      # TODO check what happens if I remove this.
                                      nn.Linear(self.input_size * 2, n_class))

    def forward(self, z_, index, batch_):
        z = z_.to(dev2)  # #nodes x 1024

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
        batch = torch.transpose(batch, 0, 1)  # 256 x #langs x 1024

        h = self.transformer(batch)  # #nodes x 1024
        h = torch.transpose(h, 0, 1)
        h = h[transformer_indices[0], transformer_indices[1],
            :]  # rearrange the nodes back to the order in which we recieved (the order that represents the graph)

        res = self.transfer(h)  # #nodes x #tags

        return res.to(dev)
