import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ------------------------
# Positional Encoding
# ------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        i = torch.arange(d_model).unsqueeze(0)
        angle_rates = 1 / torch.pow(10000, (2 * (i // 2)) / d_model)
        angle_rads = position * angle_rates

        PE = torch.zeros_like(angle_rads)
        PE[:, 0::2] = torch.sin(angle_rads[:, 0::2])
        PE[:, 1::2] = torch.cos(angle_rads[:, 1::2])

        self.register_buffer('PE', PE)

    def forward(self, x):
        seq_len = x.size(0)
        return x + self.PE[:seq_len]


# ------------------------
# Encoder Block
# ------------------------
class EncoderBlock(nn.Module):
    def __init__(self, d_model=512, num_heads=8, ffn_hidden=2048):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_Q = nn.ModuleList([nn.Linear(d_model, self.d_k) for _ in range(num_heads)])
        self.W_K = nn.ModuleList([nn.Linear(d_model, self.d_k) for _ in range(num_heads)])
        self.W_V = nn.ModuleList([nn.Linear(d_model, self.d_k) for _ in range(num_heads)])
        self.W_O = nn.Linear(d_model, d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_hidden),
            nn.ReLU(),
            nn.Linear(ffn_hidden, d_model)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        heads = []
        for i in range(self.num_heads):
            Q = self.W_Q[i](x)
            K = self.W_K[i](x)
            V = self.W_V[i](x)

            scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)
            attn = F.softmax(scores, dim=-1)
            Z = attn @ V
            heads.append(Z)

        concat = torch.cat(heads, dim=-1)
        attn_out = self.W_O(concat)
        x = self.norm1(x + attn_out)

        ffn_out = self.ffn(x)
        out = self.norm2(x + ffn_out)
        return out


# ------------------------
# Transformer Encoder Stack
# ------------------------
class TransformerEncoder(nn.Module):
    def __init__(self, num_layers=8, d_model=512, num_heads=8, ffn_hidden=2048):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderBlock(d_model, num_heads, ffn_hidden) for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# ------------------------
# Decoder Block
# ------------------------
class DecoderBlock(nn.Module):
    def __init__(self, d_model=512, num_heads=8, ffn_hidden=2048):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_Q_self = nn.ModuleList([nn.Linear(d_model, self.d_k) for _ in range(num_heads)])
        self.W_K_self = nn.ModuleList([nn.Linear(d_model, self.d_k) for _ in range(num_heads)])
        self.W_V_self = nn.ModuleList([nn.Linear(d_model, self.d_k) for _ in range(num_heads)])

        self.W_Q_encdec = nn.ModuleList([nn.Linear(d_model, self.d_k) for _ in range(num_heads)])
        self.W_K_encdec = nn.ModuleList([nn.Linear(d_model, self.d_k) for _ in range(num_heads)])
        self.W_V_encdec = nn.ModuleList([nn.Linear(d_model, self.d_k) for _ in range(num_heads)])

        self.W_O = nn.Linear(d_model, d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_hidden),
            nn.ReLU(),
            nn.Linear(ffn_hidden, d_model)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, encoder_output, look_ahead_mask=None):
        heads = []
        for i in range(self.num_heads):
            Q = self.W_Q_self[i](x)
            K = self.W_K_self[i](x)
            V = self.W_V_self[i](x)
            scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)
            if look_ahead_mask is not None:
                scores = scores.masked_fill(look_ahead_mask == 0, float('-inf'))
            attn = F.softmax(scores, dim=-1)
            Z = attn @ V
            heads.append(Z)

        concat = torch.cat(heads, dim=-1)
        self_attn_out = self.W_O(concat)
        x = self.norm1(x + self_attn_out)

        heads = []
        for i in range(self.num_heads):
            Q = self.W_Q_encdec[i](x)
            K = self.W_K_encdec[i](encoder_output)
            V = self.W_V_encdec[i](encoder_output)
            scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)
            attn = F.softmax(scores, dim=-1)
            Z = attn @ V
            heads.append(Z)

        concat = torch.cat(heads, dim=-1)
        encdec_out = self.W_O(concat)
        x = self.norm2(x + encdec_out)

        ffn_out = self.ffn(x)
        out = self.norm3(x + ffn_out)
        return out


# ------------------------
# Transformer Decoder Stack
# ------------------------
class TransformerDecoder(nn.Module):
    def __init__(self, num_layers=8, d_model=512, num_heads=8, ffn_hidden=2048):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderBlock(d_model, num_heads, ffn_hidden) for _ in range(num_layers)
        ])

    def forward(self, x, encoder_output, look_ahead_mask=None):
        for layer in self.layers:
            x = layer(x, encoder_output, look_ahead_mask)
        return x


# ------------------------
# Full Transformer Model
# ------------------------
class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_heads=8, ffn_hidden=2048, num_layers=8, max_len=5000):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.encoder = TransformerEncoder(num_layers, d_model, num_heads, ffn_hidden)
        self.decoder = TransformerDecoder(num_layers, d_model, num_heads, ffn_hidden)
        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, src_tokens, tgt_tokens, look_ahead_mask=None):
        src_embed = self.positional_encoding(self.token_embedding(src_tokens))
        tgt_embed = self.positional_encoding(self.token_embedding(tgt_tokens))
        encoder_output = self.encoder(src_embed)
        decoder_output = self.decoder(tgt_embed, encoder_output, look_ahead_mask)
        return self.output_layer(decoder_output)


# ------------------------
# Transformer With Separate Embeddings
# ------------------------
class TransformerWithEmbeddings(nn.Module):
    def __init__(self, vocab_size_src, vocab_size_tgt, d_model=512, num_heads=8, ffn_hidden=2048, num_layers=6, max_len=100):
        super().__init__()
        self.d_model = d_model
        self.src_embedding = nn.Embedding(vocab_size_src, d_model)
        self.tgt_embedding = nn.Embedding(vocab_size_tgt, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.transformer = Transformer(
            vocab_size=max(vocab_size_src, vocab_size_tgt),
            d_model=d_model,
            num_heads=num_heads,
            ffn_hidden=ffn_hidden,
            num_layers=num_layers,
            max_len=max_len
        )
        self.output_linear = nn.Linear(d_model, vocab_size_tgt)

    def forward(self, src_ids, tgt_ids):
        src = self.pos_encoding(self.src_embedding(src_ids) * math.sqrt(self.d_model))
        tgt = self.pos_encoding(self.tgt_embedding(tgt_ids) * math.sqrt(self.d_model))
        look_ahead_mask = self.generate_look_ahead_mask(tgt.size(0))
        output = self.transformer(src, tgt, look_ahead_mask)
        return self.output_linear(output)

    def generate_look_ahead_mask(self, size):
        return torch.tril(torch.ones(size, size)).bool()
