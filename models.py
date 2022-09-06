import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


def softXEnt(input, target):
    logprobs = F.log_softmax(input, dim=1)
    return -(target * logprobs).sum() / input.shape[0]

class ProtoNetItemUserLL(nn.Module):
    """
    ProtoNet for Recommendation. (Few-shot learning for new users).
    This model take users as classes and items as "images" in the original Prototype Network scene.
    """

    def __init__(self, embed_dim_item, embed_dim_user, n_users, n_items, n_memory, device,
                 model_variant='memory',
                 base_model='bpr', user_encoder=None,
                 user_embedding=None, item_embedding=None,
                 mode='base',
                 loss_type='multi',
                 dropout=0.3):
        """
        This network creats a "user embedding" for each sample
        embed_dim_user = num_query_tr
        """
        super(ProtoNetItemUserLL, self).__init__()
        self.embed_dim_item = embed_dim_item
        self.embed_dim_user = embed_dim_user
        self.n_memory = n_memory
        self.n_items = n_items
        self.device = device
        self.n_users = n_users
        self.m = nn.Tanh()
        self.gate_dropout = nn.Dropout(0.3)
        self.act = nn.PReLU()
        self.memory_list = torch.arange(n_memory).to(self.device, non_blocking=True)
        self.base_model = base_model
        self.dropout = nn.Dropout(dropout)
        self.mode = mode
        self.loss_type = loss_type
        self.few_shot_scoring = 'cos'

        self.activation = nn.LeakyReLU()
        if self.mode == 'base':
            self.item_lookup = nn.Sequential(
                nn.Embedding(n_items, embed_dim_item),
                nn.Dropout(dropout)
            )
            torch.nn.init.xavier_uniform_(self.item_lookup[0].weight)
            if base_model == 'bpr':
                self.user_lookup = nn.Sequential(
                    nn.Embedding(n_users, embed_dim_user),
                    nn.Dropout(dropout)
                )
                torch.nn.init.xavier_uniform_(self.user_lookup[0].weight)  # TODO: why not?
            elif base_model == 'vae':
                self.user_encoder = VAE(n_items, dropout=dropout)
            elif base_model == 'dae':
                self.user_encoder = DAE(n_items, n_users, dropout=dropout)
            else:
                raise NotImplementedError("base_model {} implementation not found".format(base_model))
        elif self.mode == 'few-shot':
            """Few-shot model settings"""
            if base_model == 'vae' or base_model == 'dae':
                self.user_encoder = user_encoder
            self.model_variant = model_variant
            self.memory = nn.Sequential(
                nn.Embedding(n_memory, embed_dim_item),
                nn.Dropout(dropout)
            )
            torch.nn.init.normal_(self.memory[0].weight)
            self.memory_index = nn.Sequential(
                nn.Embedding(n_memory, embed_dim_item),
                # nn.Dropout(dropout)
            )

            self.memory_index_distance = nn.Sequential(
                nn.Embedding(n_memory, embed_dim_item),
                # nn.Dropout(dropout)
            )

            torch.nn.init.xavier_uniform_(self.memory_index[0].weight)
            torch.nn.init.xavier_uniform_(self.memory_index_distance[0].weight)
            self.transform = nn.Sequential(
                nn.Linear(embed_dim_item, embed_dim_item),
                #   nn.Dropout(dropout)
            )
            self.transform1 = nn.Sequential(
                nn.Linear(embed_dim_item, embed_dim_item),
            )
            self.transform2 = nn.Sequential(
                nn.Linear(embed_dim_item * 2, embed_dim_item),
                # nn.Dropout(dropout)
            )
            self.transform_support = nn.Linear(embed_dim_item * 2, embed_dim_item)
            self.transform_pro = nn.Sequential(nn.Linear(embed_dim_item, embed_dim_item), nn.Tanh(),
                                               nn.Dropout(dropout), nn.Linear(
                    embed_dim_item, embed_dim_item))


            self.item_lookup = nn.Sequential(
                nn.Embedding.from_pretrained(item_embedding, freeze=True),
            )

            self.user_lookup = nn.Sequential(
                nn.Embedding.from_pretrained(user_embedding, freeze=False),
                nn.Dropout(dropout)
            )

            self.user_lookup_base = nn.Sequential(
                nn.Embedding.from_pretrained(user_embedding, freeze=True),
            )
            self.item_range = torch.arange(item_embedding.shape[0]).to(self.device, non_blocking=True)
            self.cos_attention = nn.CosineSimilarity(dim=3)

        self.attention = nn.Linear(embed_dim_item, 1)
        self.softmax = nn.Softmax(dim=0)
        self.weight_softmax = nn.Softmax(dim=1)
        self.cos = nn.CosineSimilarity(dim=2)

    def forward(self, support_indices, query_indices, item_idx, ui_sp_mat=None,
                pos_neg=None, mask=None):
        """
        item_indices: shape (num_query_tr, ), a list of items sampled from user
        """
        if self.base_model is not None and self.mode == 'base':
            return self.base_forward(support_indices, query_indices, item_idx, ui_sp_mat)
        else:
            return self.memory_forward(support_indices, query_indices, item_idx, ui_sp_mat, pos_neg,
                                       mask)

    def prototype(self, support_users_embed, mask=None, dimension=1):
        if mask is None:
            return torch.mean(support_users_embed, dim=dimension)
        else:
            return torch.sum(support_users_embed * mask.unsqueeze(-1), dim=dimension) / torch.sum(mask,
                                                                                                  dim=dimension).unsqueeze(
                -1)

    def memory_embedding(self, query_embed):
        n_query = query_embed.shape[0]
        cos_similarity = self.cos(self.transform(query_embed).unsqueeze(1).repeat(1, self.n_memory, 1),
                                  self.memory_index(self.memory_list).unsqueeze(0).repeat(
                                      n_query, 1, 1))  # [B, n_memory, D] -> [B, n_memory]

        attention_weights = self.weight_softmax(cos_similarity).unsqueeze(-1)  # [B, n_memory, n_heads] or [B, n_memory]


        memory_embedding = torch.sum(torch.mul(attention_weights, self.memory(self.memory_list)), dim=1)

        gating = self.gate_dropout(self.m(self.transform2(torch.cat((memory_embedding, query_embed), dim=1))))

        return torch.mul(1 - gating, self.gate_dropout(query_embed)) + torch.mul(gating, self.gate_dropout(
            memory_embedding)), memory_embedding

    def loss_function(self, logits, targets, weight):
        n_classes = logits.shape[-1]
        batch_size = logits.shape[0]
        num_ex = logits.view(batch_size, -1).shape[1]
        if self.loss_type == 'multi' or (self.mode == 'base' and (self.base_model == 'vae' or self.base_model == 'dae')):
            return -torch.mean(
                torch.sum(F.log_softmax(logits.view(-1, n_classes), 1) * targets.view(-1, n_classes), -1))
        elif self.loss_type == 'bce':
            return F.binary_cross_entropy_with_logits(logits.view(batch_size, -1), targets.view(batch_size, -1),
                                                      pos_weight=weight.view(num_ex))
        else:
            raise NotImplementedError("loss_type {} implementation not found".format(self.loss_type))

    def base_forward(self, support_indices, query_indices, item_idx, ui_sp_mat):
        query_indices = torch.cat((support_indices, query_indices), dim=1)
        assert self.base_model is not None
        if self.base_model == 'bpr':
            query_items_embed = self.user_lookup(query_indices).view(-1, query_indices.shape[1],
                                                                     self.embed_dim_item)  # [B, K, D]
            support_centers = self.item_lookup(torch.arange(self.n_items).to(self.device, non_blocking=True))
        elif self.base_model == 'vae' or self.base_model == 'dae':
            query_items_embed, _, _ = self.user_encoder(
                torch.Tensor(ui_sp_mat[query_indices.view(-1).cpu().numpy(), :].toarray()).squeeze().to(self.device,
                                                                                                        non_blocking=True),
                query_indices.view(-1))
            query_items_embed = query_items_embed.view(-1, query_indices.shape[1],
                                                       self.embed_dim_item)  # TODO: query = support + query
            support_centers = self.item_lookup(torch.arange(self.n_items).to(self.device, non_blocking=True))  # [I * D]
        else:
            raise NotImplementedError("base_model {} implementation not found".format(self.base_model))

        batch_size = item_idx.shape[0]

        num_query = query_items_embed.shape[1]

        if self.base_model == 'bpr':
            query_vecs = query_items_embed.view(-1, self.embed_dim_user)
            relation_scores = torch.mm(F.normalize(query_vecs, dim=-1),
                                       F.normalize(support_centers, dim=-1).permute(1, 0)).view(batch_size, num_query,
                                                                                                self.n_items) * 10.0
        elif self.base_model == 'vae' or self.base_model == 'dae':
            query_vecs = query_items_embed.view(-1, self.embed_dim_user)
            relation_scores = torch.mm(F.normalize(query_vecs, dim=-1),
                                       F.normalize(support_centers, dim=-1).permute(1, 0)).view(batch_size, num_query,
                                                                                                self.n_items) * 10.0
        # [B, K, B]

        return relation_scores, None

    def support_gating(self, support_items_embed, support_centers):
        support_centers = support_centers.unsqueeze(1).repeat(1, support_items_embed.shape[1], 1)
        gating = self.transform_support(torch.cat((support_centers, support_items_embed), dim=2))
        new_embed = torch.mul(1 - gating, self.gate_dropout(support_centers)) + torch.mul(gating, self.gate_dropout(
            support_items_embed))

        return self.prototype(new_embed)

    def memory_forward(self, support_indices, query_indices, item_idx, ui_sp_mat, pos_neg, mask):
        n_support = support_indices.shape[1]
        # user embedding lookup.
        if self.base_model == 'bpr':
            support_items_embed = self.user_lookup(support_indices)
            query_items_embed = self.user_lookup(query_indices)  # [B, K, D]
        elif self.base_model == 'vae' or self.base_model == 'dae':
            support_items_embed, _, _ = self.user_encoder(
                torch.Tensor(ui_sp_mat[support_indices.view(-1).cpu().numpy(), :].toarray()).squeeze().to(self.device,
                                                                                                          non_blocking=True),
                support_indices.view(-1))
            support_items_embed = support_items_embed.view(-1, n_support, self.embed_dim_item)

            query_items_embed, _, _ = self.user_encoder(
                torch.Tensor(ui_sp_mat[query_indices.view(-1).cpu().numpy(), :].toarray()).squeeze().to(self.device,
                                                                                                        non_blocking=True),
                query_indices.view(-1))
            query_items_embed = query_items_embed.view(-1, query_indices.shape[1], self.embed_dim_item)

        support_centers = self.prototype(support_items_embed, mask=mask)  # [B, D]
        support_centers, memory_embed = self.memory_embedding(support_centers)

        batch_size = support_centers.shape[0]
        num_query = query_items_embed.shape[1]
        query_vecs = query_items_embed.view(-1, self.embed_dim_user).unsqueeze(1).repeat(1, batch_size,
                                                                                         1)  # [B * K, B, D]
        support_vecs = support_centers.unsqueeze(0).repeat(query_vecs.shape[0], 1, 1)  # [B * K, B, D]
        # Compute distance from each of B*K to each centroid.

        relation_scores = self.cos(support_vecs, query_vecs).view(batch_size, num_query,
                                                                  batch_size) * 10
        positive_embed = memory_embed[pos_neg[0]]
        negative_embed = memory_embed[pos_neg[1]]  # B * 5 * D
        curr_item = memory_embed.unsqueeze(1)

        pos_dist = ((curr_item - positive_embed) ** 2).sum(-1)
        neg_dist = ((curr_item - negative_embed) ** 2).sum(-1)
        logits = torch.cat((- pos_dist, - neg_dist), dim=1)  # [B, pos + neg]
        targets = torch.cat((torch.ones_like(pos_dist), torch.zeros_like(neg_dist)), dim=1)  # [B, pos + neg]
        n_classes = logits.shape[-1]

        correction = -torch.mean(
            torch.sum(F.log_softmax(logits.view(-1, n_classes), 1) * targets.view(-1, n_classes), -1))

        return relation_scores, correction


class VAE(nn.Module):
    """
    Container module for Multi-VAE.
    Multi-VAE : Variational Autoencoder with Multinomial Likelihood
    See Variational Autoencoders for Collaborative Filtering
    https://arxiv.org/abs/1802.05814
    """

    def __init__(self, n_items, dropout=0.5):
        super(VAE, self).__init__()

        self.enc_dims = [n_items, 256, 128]

        # Last dimension of q- network is for mean and variance
        temp_q_dims = self.enc_dims[:-1] + [self.enc_dims[-1] * 2]
        self.enc_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
                                         d_in, d_out in zip(temp_q_dims[:-1], temp_q_dims[1:])])

        self.drop = nn.Dropout(dropout)
        self.init_weights()

    def forward(self, input, idx):
        if len(input.shape) < 2:
            input = input.unsqueeze(0)
        mu, logvar = self.encode(input)  # Encoder computes mean and log variance.
        z = self.reparameterize(mu, logvar)
        return z, None, None

    def encode(self, input):
        h = F.normalize(input)
        h = self.drop(h)

        for i, layer in enumerate(self.enc_layers):
            h = layer(h)
            if i != len(self.enc_layers) - 1:
                h = torch.tanh(h)
            else:
                mu = h[:, :self.enc_dims[-1]]
                logvar = h[:, self.enc_dims[-1]:]
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def init_weights(self):
        for layer in self.enc_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)


class DAE(nn.Module):
    """
    Container module for Multi-DAE.
    Multi-DAE : Denoising Autoencoder with Multinomial Likelihood
    See Variational Autoencoders for Collaborative Filtering
    https://arxiv.org/abs/1802.05814
    """

    def __init__(self, n_items, n_users, dropout=0.5):
        super(DAE, self).__init__()

        self.enc_dims = [n_items, 256, 128]

        # Last dimension of q- network is for mean and variance
        self.dims = self.enc_dims
        self.layers = nn.ModuleList([nn.Linear(d_in, d_out) for
                                     d_in, d_out in zip(self.dims[:-1], self.dims[1:])])
        self.drop = nn.Dropout(dropout)
        self.user_lookup = nn.Sequential(nn.Embedding(n_users, 128), nn.Dropout(dropout))
        torch.nn.init.xavier_uniform_(self.user_lookup[0].weight)
        self.init_weights()

    def forward(self, input, idx):
        if len(input.shape) < 2:
            input = input.unsqueeze(0)
        h = F.normalize(input)
        h = self.drop(h)
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i == 1:
                h += self.user_lookup(idx.squeeze())
            if i != len(self.layers) - 1:
                h = torch.tanh(h)
        return h, None, None

    def init_weights(self):
        for layer in self.layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)
