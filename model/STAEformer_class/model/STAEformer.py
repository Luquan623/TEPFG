import torch.nn as nn
import torch



class AttentionLayer(nn.Module):
    """Perform attention across the -2 dim (the -1 dim is `model_dim`).

    Make sure the tensor is permuted to correct shape before attention.

    E.g.
    - Input shape (batch_size, in_steps, num_nodes, model_dim).
    - Then the attention will be performed across the nodes.

    Also, it supports different src and tgt length.

    But must `src length == K length == V length`.

    """

    def __init__(self, model_dim, num_heads=8, mask=False):
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask

        self.head_dim = model_dim // num_heads

        self.FC_Q = nn.Linear(model_dim, model_dim)
        self.FC_K = nn.Linear(model_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)

        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, query, key, value):
        # Q    (batch_size, ..., tgt_length, model_dim)
        # K, V (batch_size, ..., src_length, model_dim)
        batch_size = query.shape[0]
        tgt_length = query.shape[-2]
        src_length = key.shape[-2]

        query = self.FC_Q(query)
        key = self.FC_K(key)
        value = self.FC_V(value)

        # Qhead, Khead, Vhead (num_heads * batch_size, ..., length, head_dim) 多头
        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)

        key = key.transpose(
            -1, -2
        )  # (num_heads * batch_size, ..., head_dim, src_length)

        attn_score = (
            query @ key
        ) / self.head_dim**0.5  # (num_heads * batch_size, ..., tgt_length, src_length)

        if self.mask:
            mask = torch.ones(
                tgt_length, src_length, dtype=torch.bool, device=query.device
            ).tril()  # lower triangular part of the matrix
            attn_score.masked_fill_(~mask, -torch.inf)  # fill in-place

        attn_score = torch.softmax(attn_score, dim=-1)
        out = attn_score @ value  # (num_heads * batch_size, ..., tgt_length, head_dim)
        out = torch.cat(
            torch.split(out, batch_size, dim=0), dim=-1
        )  # (batch_size, ..., tgt_length, head_dim * num_heads = model_dim)

        out = self.out_proj(out)

        return out


class SelfAttentionLayer(nn.Module):
    def __init__(
        self, model_dim, feed_forward_dim=2048, num_heads=8, dropout=0, mask=False
    ):
        super().__init__()

        self.attn = AttentionLayer(model_dim, num_heads, mask)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, dim=-2):
        x = x.transpose(dim, -2)
        # x: (batch_size, ..., length, model_dim)
        residual = x
        out = self.attn(x, x, x)  # (batch_size, ..., length, model_dim)
        out = self.dropout1(out)
        out = self.ln1(residual + out)

        residual = out
        out = self.feed_forward(out)  # (batch_size, ..., length, model_dim)
        out = self.dropout2(out)
        out = self.ln2(residual + out)

        out = out.transpose(dim, -2)
        return out


class Model(nn.Module):
    def __init__(
        self,
        opt
    ):
        super().__init__()

        self.num_nodes = opt.num_nodes
        self.in_steps = opt.in_steps
        self.out_steps = opt.out_steps
        self.steps_per_day = opt.steps_per_day
        self.input_dim = opt.input_dim
        self.output_dim = opt.output_dim
        self.input_embedding_dim = opt.input_embedding_dim
        self.tod_embedding_dim = opt.tod_embedding_dim
        self.dow_embedding_dim = opt.dow_embedding_dim
        self.spatial_embedding_dim = opt.spatial_embedding_dim
        self.adaptive_embedding_dim = opt.adaptive_embedding_dim
        self.feed_forward_dim = opt.feed_forward_dim
        self.dropout = opt.dropout
        self.model_dim = (
            self.input_embedding_dim
            + self.tod_embedding_dim
            + self.dow_embedding_dim
            + self.spatial_embedding_dim
            + self.adaptive_embedding_dim
        )
        self.num_heads = opt.num_heads
        self.num_layers = opt.num_layers


        self.input_proj = nn.Linear(self.input_dim, self.input_embedding_dim)
        if opt.tod_embedding_dim > 0:
            self.tod_embedding = nn.Embedding(self.steps_per_day, self.tod_embedding_dim) # 每个时间点会被映射成24维的嵌入向量
        if opt.dow_embedding_dim > 0:
            self.dow_embedding = nn.Embedding(7, self.dow_embedding_dim)
        if opt.spatial_embedding_dim > 0:
            self.node_emb = nn.Parameter(
                torch.empty(self.num_nodes, self.spatial_embedding_dim)
            )
            nn.init.xavier_uniform_(self.node_emb)
        if opt.adaptive_embedding_dim > 0:
            # self.adaptive_embedding = nn.init.xavier_uniform_(
            #     nn.Parameter(torch.empty(self.in_steps, self.num_nodes, self.adaptive_embedding_dim))
            # )

            self.adaptive_embedding = nn.Parameter(
                torch.empty(self.in_steps, self.num_nodes, self.adaptive_embedding_dim)
            )
            nn.init.xavier_uniform_(self.adaptive_embedding)

        # if self.use_mixed_proj:
        #     self.output_proj = nn.Linear(
        #         self.in_steps * self.model_dim, self.out_steps * self.output_dim
        #     )
        # else:
        self.temporal_proj = nn.Linear(self.in_steps, self.out_steps)

        # 替换 output_proj 为分类头：MLP + Sigmoid（或配合 BCEWithLogitsLoss）
        self.classifier = nn.Sequential(
            nn.Linear(self.model_dim, 1)  # 只输出一个 logit 表示极端值概率
            # 不加 sigmoid，训练时使用 BCEWithLogitsLoss
        )


        self.attn_layers_t = nn.ModuleList(
            [
                SelfAttentionLayer(self.model_dim, self.feed_forward_dim, self.num_heads, self.dropout)
                for _ in range(self.num_layers)
            ]
        )

        self.attn_layers_s = nn.ModuleList(
            [
                SelfAttentionLayer(self.model_dim, self.feed_forward_dim, self.num_heads, self.dropout)
                for _ in range(self.num_layers)
            ]
        )

    def forward(self, x):
        # x: (batch_size, in_steps, num_nodes, input_dim+tod+dow=3)
        batch_size = x.shape[0]

        if self.tod_embedding_dim > 0:
            tod = x[..., 1]
        if self.dow_embedding_dim > 0:
            dow = x[..., 2]

        x = x[..., : self.input_dim]  # (8,12,307,4)
        #x = x[..., -1: ]

        x = self.input_proj(x)   # (batch_size, in_steps, num_nodes, input_embedding_dim),将4维特征映射成24维 (8,12,307,24)
        features = [x]
        if self.tod_embedding_dim > 0:
            tod_emb = self.tod_embedding(
                (tod * self.steps_per_day).long()
            )  # (batch_size, in_steps, num_nodes, tod_embedding_dim) # tod * self.steps_per_day 就把归一化时间转成一个具体的“时间步索引”，.long() 是为了变成整数索引，用于查嵌入表，最终 self.tod_embedding(...) 是一个 nn.Embedding 层，查出这个时间步对应的嵌入向量
            features.append(tod_emb)
        if self.dow_embedding_dim > 0:
            dow_emb = self.dow_embedding(
                dow.long()
            )  # (batch_size, in_steps, num_nodes, dow_embedding_dim)
            features.append(dow_emb)
        if self.spatial_embedding_dim > 0: # 节点嵌入，用时空嵌入代替了
            spatial_emb = self.node_emb.expand(
                batch_size, self.in_steps, *self.node_emb.shape
            )
            features.append(spatial_emb)
        if self.adaptive_embedding_dim > 0: # 时空嵌入，既包含时间也包含空间
            adp_emb = self.adaptive_embedding.expand(
                size=(batch_size, *self.adaptive_embedding.shape)
            )
            features.append(adp_emb)
        x = torch.cat(features, dim=-1)  # (batch_size, in_steps, num_nodes, model_dim)

        for attn in self.attn_layers_t:
            x = attn(x, dim=1)
        for attn in self.attn_layers_s:
            x = attn(x, dim=2)
        # (batch_size, in_steps, num_nodes, model_dim)

        # if self.use_mixed_proj:
        #     out = x.transpose(1, 2)  # (batch_size, num_nodes, in_steps, model_dim)
        #     out = out.reshape(
        #         batch_size, self.num_nodes, self.in_steps * self.model_dim
        #     )
        #     out = self.output_proj(out).view(
        #         batch_size, self.num_nodes, self.out_steps, self.output_dim
        #     )
        #     out = out.transpose(1, 2)  # (batch_size, out_steps, num_nodes, output_dim)

        out = x.transpose(1, 3)  # (B, model_dim, N, T)
        out = self.temporal_proj(out)  # (B, model_dim, N, out_steps)
        out = out.transpose(1, 3)  # (B, out_steps, N, model_dim)
        out = self.classifier(out)  # (B, out_steps, N, 1)

        return out

