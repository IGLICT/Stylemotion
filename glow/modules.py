# import torch
# import torch.nn as nn
# import torch.nn.functional as F

import jittor as jt
from jittor import init
from jittor import nn

import numpy as np
import scipy.linalg
from . import thops
import math

def nan_throw(tensor, name="tensor"):
    stop = False
    # if ((tensor != tensor).any()):
    #     print(name + " has nans")
    #     stop = True
    # if (torch.isinf(tensor).any()):
    #     print(name + " has infs")
    #     stop = True
    if stop:
        print(name + ": " + str(tensor))
        # raise ValueError(name + ' contains nans of infs')


class _ActNorm(nn.Module):
    """
    Activation Normalization
    Initialize the bias and scale with a given minibatch,
    so that the output per-channel have zero mean and unit variance for that.

    After initialization, `bias` and `logs` will be trained as parameters.
    """

    def __init__(self, num_features, scale=1.):
        super().__init__()
        # register mean and scale
        size = [1, num_features, 1]
        # self.register_parameter("bias", nn.Parameter(torch.zeros(*size)))
        # self.register_parameter("logs", nn.Parameter(torch.zeros(*size)))
        # self.bias = jt.zeros(size)
        # self.logs = jt.zeros(size)

        self.bias = jt.zeros((1, num_features, 1))
        self.logs = jt.zeros((1, num_features, 1))
        self.num_features = num_features
        self.scale = float(scale)
        self.inited = False

    def _check_input_dim(self, input):
        return NotImplemented

    import copy
    def initialize_parameters(self, input):
        self._check_input_dim(input)
        # if not self.training:
        #     return
        # assert input.device == self.bias.device
        with jt.no_grad():
            bias = thops.mean(copy.deepcopy(input), dim=[0, 2], keepdim=True) * -1.0
            vars = thops.mean((copy.deepcopy(input) + bias) ** 2, dim=[0, 2], keepdim=True)
            logs = jt.log(self.scale / (jt.sqrt(vars) + 1e-6))        
        self.bias.data = bias.data
        self.logs.data = logs.data
        self.inited = True

    def _center(self, input, reverse=False):
        if not reverse:
            return input + self.bias
        else:
            return input - self.bias

    def _scale(self, input, logdet=None, reverse=False):
        logs = self.logs
        if not reverse:
            input = input * jt.exp(logs)
        else:
            input = input * jt.exp(-logs)
        if logdet is not None:
            """
            logs is log_std of `mean of channels`
            so we need to multiply timesteps
            """
            dlogdet = thops.sum(logs) * thops.timesteps(input)
            if reverse:
                dlogdet *= -1
            logdet = logdet + dlogdet
        return input, logdet

    def execute(self, input, logdet=None, reverse=False):
        if not self.inited:
            self.initialize_parameters(input)
        self._check_input_dim(input)
        # no need to permute dims as old version
        if not reverse:
            # center and scale
            input = self._center(input, reverse)
            input, logdet = self._scale(input, logdet, reverse)
        else:
            # scale and center
            input, logdet = self._scale(input, logdet, reverse)
            input = self._center(input, reverse)
        return input, logdet


class ActNorm2d(_ActNorm):
    def __init__(self, num_features, scale=1.):
        super().__init__(num_features, scale)

    def _check_input_dim(self, input):
        assert len(input.shape) == 3
        # assert input.size(1) == self.num_features, (
        #     "[ActNorm]: input should be in shape as `BCT`,"
        #     " channels should be {} rather than {}".format(
        #         self.num_features, input.size()))


class LinearZeros(nn.Linear):
    def __init__(self, in_channels, out_channels, logscale_factor=3):
        super().__init__(in_channels, out_channels)
        self.logscale_factor = logscale_factor
        # set logs parameter
        self.register_parameter("logs", nn.Parameter(torch.zeros(out_channels)))
        # init
        self.weight.data.zero_()
        self.bias.data.zero_()

    def forward(self, input):
        output = super().forward(input)
        return output * torch.exp(self.logs * self.logscale_factor)


class Conv2d(nn.Conv2d):
    pad_dict = {
        "same": lambda kernel, stride: [((k - 1) * s + 1) // 2 for k, s in zip(kernel, stride)],
        "valid": lambda kernel, stride: [0 for _ in kernel]
    }

    @staticmethod
    def get_padding(padding, kernel_size, stride):
        # make paddding
        if isinstance(padding, str):
            if isinstance(kernel_size, int):
                kernel_size = [kernel_size, kernel_size]
            if isinstance(stride, int):
                stride = [stride, stride]
            padding = padding.lower()
            try:
                padding = Conv2d.pad_dict[padding](kernel_size, stride)
            except KeyError:
                raise ValueError("{} is not supported".format(padding))
        return padding

    def __init__(self, in_channels, out_channels,
                 kernel_size=[3, 3], stride=[1, 1],
                 padding="same", do_actnorm=True, weight_std=0.05):
        padding = Conv2d.get_padding(padding, kernel_size, stride)
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, bias=(not do_actnorm))
        # init weight with std
        self.weight.data.normal_(mean=0.0, std=weight_std)
        if not do_actnorm:
            self.bias.data.zero_()
        else:
            self.actnorm = ActNorm2d(out_channels)
        self.do_actnorm = do_actnorm

    def forward(self, input):
        x = super().forward(input)
        if self.do_actnorm:
            x, _ = self.actnorm(x)
        return x


class GCN(nn.Conv2d):
    pad_dict = {
        "same": lambda kernel, stride: [((k - 1) * s + 1) // 2 for k, s in zip(kernel, stride)],
        "valid": lambda kernel, stride: [0 for _ in kernel]
    }

    @staticmethod
    def get_padding(padding, kernel_size, stride):
        # make paddding
        if isinstance(padding, str):
            if isinstance(kernel_size, int):
                kernel_size = [kernel_size, kernel_size]
            if isinstance(stride, int):
                stride = [stride, stride]
            padding = padding.lower()
            try:
                padding = Conv2d.pad_dict[padding](kernel_size, stride)
            except KeyError:
                raise ValueError("{} is not supported".format(padding))
        return padding

    def __init__(self, in_channels, out_channels,
                 kernel_size=[3, 3], stride=[1, 1],
                 padding="same", do_actnorm=True, weight_std=0.05):
        padding = Conv2d.get_padding(padding, kernel_size, stride)
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, bias=(not do_actnorm))
        # init weight with std
        self.weight.data.normal_(mean=0.0, std=weight_std)
        if not do_actnorm:
            self.bias.data.zero_()
        else:
            self.actnorm = ActNorm2d(out_channels)
        self.do_actnorm = do_actnorm

    def forward(self, input):
        x = super().forward(input)
        if self.do_actnorm:
            x, _ = self.actnorm(x)
        return x


class Conv2dZeros(nn.Conv2d):
    def __init__(self, in_channels, out_channels,
                 kernel_size=[3, 3], stride=[1, 1],
                 padding="same", logscale_factor=3):
        padding = Conv2d.get_padding(padding, kernel_size, stride)
        super().__init__(in_channels, out_channels, kernel_size, stride, padding)
        # logscale_factor
        self.logscale_factor = logscale_factor
        self.register_parameter("logs", nn.Parameter(torch.zeros(out_channels, 1, 1)))
        # init
        self.weight.data.zero_()
        self.bias.data.zero_()

    def forward(self, input):
        output = super().forward(input)
        return output * torch.exp(self.logs * self.logscale_factor)


class LinearNormInit(nn.Linear):
    def __init__(self, in_channels, out_channels, weight_std=0.05):
        super().__init__(in_channels, out_channels)
        # init
        self.weight.data.normal_(mean=0.0, std=weight_std)
        self.bias.data.zero_()


class LinearZeroInit(nn.Linear):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels)
        # init
        # self.weight.data.zero_()
        # self.bias.data.zero_()


class Permute2d(nn.Module):
    def __init__(self, num_channels, shuffle):
        super().__init__()
        self.num_channels = num_channels
        print(num_channels)
        self.indices = np.arange(self.num_channels - 1, -1, -1).astype(np.long)
        self.indices_inverse = np.zeros((self.num_channels), dtype=np.long)
        print(self.indices_inverse.shape)
        for i in range(self.num_channels):
            self.indices_inverse[self.indices[i]] = i
        if shuffle:
            self.reset_indices()

    def reset_indices(self):
        np.random.shuffle(self.indices)
        for i in range(self.num_channels):
            self.indices_inverse[self.indices[i]] = i
    #
    # def get_weight(self, input, reverse):
    #
    #     timesteps = thops.timesteps(input)
    #     if not reverse:
    #         dlogdet = torch.slogdet(self.indices)[1] * timesteps
    #     else:
    #         dlogdet = torch.slogdet(self.indices_inverse)[1] * timesteps
    #
    #     return dlogdet

    def forward(self, input, reverse=False):
        # weight, dlogdet = self.get_weight(input, reverse)
        assert len(input.size()) == 3
        if not reverse:
            # if logdet is not None:
            #     logdet = logdet + dlogdet
            return input[:, self.indices, :]
        else:
            # if logdet is not None:
            #     logdet = logdet - dlogdet
            return input[:, self.indices_inverse, :]


class InvertibleConv1x1(nn.Module):
    def __init__(self, num_channels, LU_decomposed=False):
        super().__init__()
        w_shape = [num_channels, num_channels]
        np.random.seed(1) # add for debug
        w_init = np.linalg.qr(np.random.randn(*w_shape))[0].astype(np.float32)
        if not LU_decomposed:
            # Sample a random orthogonal matrix:
            # self.register_parameter("weight", nn.Parameter(torch.Tensor(w_init)))
            self.weight = jt.array(w_init)
        else:
            np_p, np_l, np_u = scipy.linalg.lu(w_init)
            np_s = np.diag(np_u)
            np_sign_s = np.sign(np_s)
            np_log_s = np.log(np.abs(np_s))
            np_u = np.triu(np_u, k=1)
            l_mask = np.tril(np.ones(w_shape, dtype=np.float32), -1)
            eye = np.eye(*w_shape, dtype=np.float32)

            # self.p = torch.Tensor(np_p.astype(np.float32))
            # self.sign_s = torch.Tensor(np_sign_s.astype(np.float32))
            # self.register_buffer('p', torch.Tensor(np_p.astype(np.float32)))
            # self.register_buffer('sign_s', torch.Tensor(np_sign_s.astype(np.float32)))
            self.p = jt.array(np_p.astype(np.float32)).stop_grad()
            self.sign_s = jt.array(np_sign_s.astype(np.float32)).stop_grad()
            self.l = jt.array(np_l.astype(np.float32)).stop_grad()
            self.log_s = jt.array(np_log_s.astype(np.float32)).stop_grad()
            self.u = jt.array(np_u.astype(np.float32)).stop_grad()
            self.l_mask = jt.array(l_mask.astype(np.float32)).stop_grad()
            self.eye = jt.array(eye).stop_grad()
        self.w_shape = w_shape
        self.LU = LU_decomposed

    def get_weight(self, input, reverse):
        w_shape = self.w_shape
        if not self.LU:
            timesteps = thops.timesteps(input)
            dlogdet = jt.slogdet(self.weight)[1] * timesteps
            if not reverse:
                weight = self.weight.view(w_shape[0], w_shape[1], 1)
            else:
                weight = jt.inverse(self.weight.double()).float() \
                    .view(w_shape[0], w_shape[1], 1)
            return weight, dlogdet
        else:
            # self.p = self.p.to(input.device)
            # self.sign_s = self.sign_s.to(input.device)
            # self.l_mask = self.l_mask.to(input.device)
            # self.eye = self.eye.to(input.device)
            l = self.l * self.l_mask + self.eye
            u = self.u * self.l_mask.transpose(1, 0) + jt.diag(self.sign_s * jt.exp(self.log_s))
            dlogdet = thops.sum(self.log_s) * thops.timesteps(input)
            if not reverse:
                w = jt.matmul(self.p, jt.matmul(l, u))
            else:
                # l = jt.linalg.inv(l.astype(jt.float32))
                l.data = np.linalg.inv(l.data)
                u.data = np.linalg.inv(u.data)
                l.data = np.linalg.inv(l.data)
                w_data = np.matmul(u.data, np.matmul(l.data, np.linalg.inv(self.p.data)))
                # print("cal LU")
                w = jt.array(w_data)
                # w = jt.matmul(u, jt.matmul(l, jt.linalg.inv(self.p.astype(jt.float32))))
            return w.view(w_shape[0], w_shape[1], 1), dlogdet

    def execute(self, input, logdet=None, reverse=False):
        """
        log-det = log|abs(|W|)| * timesteps
        """
        weight, dlogdet = self.get_weight(input, reverse)
        nan_throw(weight, "weight")
        nan_throw(dlogdet, "dlogdet")

        if not reverse:
            # z = F.conv1d(input, weight)
            bs, num_f, num_seq = input.shape
            # input = input.permute(0,2,1).reshape(-1,num_f,1)
            z = nn.bmm(weight.permute(2,0,1).repeat(bs,1,1), input)
            if logdet is not None:
                logdet = logdet + dlogdet
            return z, logdet
        else:
            nan_throw(input, "InConv input")
            bs, num_f, num_seq = input.shape
            z = nn.bmm(weight.permute(2,0,1).repeat(bs,1,1), input)
            nan_throw(z, "InConv z")
            if logdet is not None:
                logdet = logdet - dlogdet
            nan_throw(logdet, "InConv logdet")
            
            return z, logdet


# Here we define our model as a class
class LSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim=1, num_layers=2, dropout=0.0):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Define the LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, batch_first=True)

        # Define the output layer
        self.linear = LinearZeroInit(self.hidden_dim, output_dim)

        # do_init
        self.do_init = True

    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        self.do_init = True

    def forward(self, input):
        # Forward pass through LSTM layer
        # shape of lstm_out: [batch_size, input_size, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both
        # have shape (batch_size, num_layers, hidden_dim).
        if self.do_init:
            lstm_out, self.hidden = self.lstm(input)
            self.do_init = False
        else:
            lstm_out, self.hidden = self.lstm(input, self.hidden)

        # self.hidden = hidden[0].to(input.device), hidden[1].to(input.device)

        # Final layer
        y_pred = self.linear(lstm_out)
        return y_pred

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = jt.zeros((max_len, d_model))
        position = jt.arange(0, max_len, dtype=jt.float).unsqueeze(1)
        div_term = jt.exp(jt.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = jt.sin(position * div_term)
        pe[:, 1::2] = jt.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(1,0,2)
        # self.register_buffer('pe', pe)
        self.pe = pe

    def execute(self, x):
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)

# class TransformerModel(nn.Module):

#     def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
#         super(TransformerModel, self).__init__()
#         from torch.nn import TransformerEncoder, TransformerEncoderLayer
#         self.model_type = 'Transformer'
#         self.src_mask = None
#         self.pos_encoder = PositionalEncoding(ninp, dropout)
#         encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
#         self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
#         # self.encoder = nn.Embedding(ntoken, ninp)
#         self.ninp = ninp
#         # self.decoder = nn.Linear(ninp, ntoken)
#         self.decoder = LinearZeroInit(ninp, ntoken)
#         # self.init_weights()

#     def _generate_square_subsequent_mask(self, sz):
#         mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
#         mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
#         return mask

#     # def init_weights(self):
#     #     initrange = 0.1
#     #     # self.encoder.weight.data.uniform_(-initrange, initrange)
#     #     self.decoder.bias.data.zero_()
#     #     self.decoder.weight.data.uniform_(-initrange, initrange)

#     def forward(self, src): # 100, 80, 34
#         src = src.permute(1, 0, 2)
#         if self.src_mask is None or self.src_mask.size(0) != src.size(0):
#             device = src.device
#             mask = self._generate_square_subsequent_mask(src.size(0)).to(device)
#             self.src_mask = mask

#         # src = self.encoder(src) * math.sqrt(self.ninp)
#         src = self.pos_encoder(src)
#         output = self.transformer_encoder(src, self.src_mask)
#         output = self.decoder(output)
#         output = output.permute(1, 0, 2)
#         return output

###########################  syt  ################################
# class multiheadAttention(nn.Module):
#     def __init__(self, embed_dim, nhead, dropout=0.1):
#         super(multiheadAttention, self).__init__()
#         self.k_dim = self.v_dim = embed_dim
#         self.embed_dim = embed_dim
#         self.num_heads = nhead
#         self.dropout = dropout
#         self.head_dim = embed_dim // self.num_heads

#         self.q_proj_weight = nn.Linear(embed_dim, embed_dim, bias=False)
#         self.k_proj_weight = nn.Linear(embed_dim, embed_dim, bias=False)
#         self.v_proj_weight = nn.Linear(embed_dim, embed_dim, bias=False)

#         self.out_proj = nn.Linear(embed_dim, self.embed_dim)
#         self.softmax = nn.Softmax(dim=-1)
#         self.act = nn.ReLU()
#         self.norm = nn.BatchNorm1d(embed_dim)
#         # pass
    
#     def execute(self, q, k, v, atten_mask):
#         # q/k/v [Num, bs, embed_dim]
#         Num, bs = q.shape[:2]

#         q = q.view(-1, self.embed_dim)
#         k = k.view(-1, self.embed_dim)
#         v = v.view(-1, self.embed_dim)

#         q_ = self.q_proj_weight(q) # [Num*bs, dim]
#         k_ = self.k_proj_weight(k)
#         v_ = self.v_proj_weight(v)
#         q_ = q_.view(Num,bs,-1).permute(1,0,2) # [bs, Num, embed_dim]
#         k_ = k_.view(Num,bs,-1).permute(1,0,2) # [bs, Num, embed_dim]
#         v_ = v_.view(Num,bs,-1).permute(1,0,2) # [bs, Num, embed_dim]

#         energy = nn.bmm(q_, k_.permute(0,2,1)) # [bs, Num, Num]
#         energy = energy * atten_mask.unsqueeze(0)
#         attention = self.softmax(energy)
#         attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))

#         x_r = nn.bmm(v_.permute(0,2,1), attention) # [bs, embed_dim, Num]
#         x_r = x_r.permute(0,2,1).view(Num*bs, -1)
#         x_r = self.act(self.norm(self.out_proj(x_r)))
#         return x_r.view(bs,Num,self.embed_dim).permute(1,0,2)

class multiheadAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.1, proj_drop=0.1):
        super(multiheadAttention,self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def execute(self, x, atten_mask): # batch_size, Num, channel
        b,n,c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c// self.num_heads).permute(2, 0, 3, 1, 4)
        
        q,k,v = qkv[0],qkv[1],qkv[2]

        attn = nn.bmm_transpose(q, k) # [100,2,22,347,] x [100,2,22,347,] -> [100,2,22,22,] 
        attn = attn + atten_mask.unsqueeze(0).unsqueeze(1)
        
        attn = nn.softmax(attn,dim=-1)

        attn = self.attn_drop(attn)

        out = nn.bmm(attn,v) # [100,2,22,22,] x [100,2,22,347,] -> [100,2,22,347,]
        out = out.transpose(0,2,1,3).reshape(b,n,c)
        out = self.proj(out)
        out = self.proj_drop(out)

        return out


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerEncoderLayer, self).__init__()
        self.self_atten = multiheadAttention(d_model, nhead, attn_drop=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.norm = nn.BatchNorm1d(d_model)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm2 = nn.BatchNorm1d(d_model)
        self.act = nn.ReLU()

    def execute(self, src, src_mask):
        src2 = self.self_atten(src, src_mask)
        src = src + src2
        src = self.norm(src.permute(0,2,1)).permute(0,2,1)
        src2 = self.linear2(self.act(self.linear1(src)))
        src = src + src2
        src = self.norm2(src.permute(0,2,1)).permute(0,2,1)
        return src

import copy
class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super(TransformerEncoder, self).__init__()
        self.layers = [copy.deepcopy(encoder_layer) for _ in range(num_layers)]
        self.num_layers = num_layers

    def execute(self, src, src_mask):
        output = src
        for mod in self.layers:
            output = mod(output, src_mask=src_mask)
        return output

class TransformerModel(nn.Module):

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        # from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        # self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        # self.decoder = nn.Linear(ninp, ntoken)
        self.decoder = LinearZeroInit(ninp, ntoken)
        # self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        if sz == 1:
            return jt.array([[0]])
        mask = (jt.misc.triu_(jt.ones((sz, sz))) == 1).transpose(1, 0).stop_grad()
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def execute(self, src): # 100, 80, 34
        # src = src.permute(1, 0, 2)
        if self.src_mask is None or self.src_mask.shape[0] != src.shape[0]:
            # device = src.device
            mask = self._generate_square_subsequent_mask(src.shape[1])
            self.src_mask = mask

        # src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        # output = output.permute(1, 0, 2)
        return output




# Here we define our model as a class
class GRU(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim=1, num_layers=2, dropout=0.0):
        super(GRU, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Define the LSTM layer
        self.gru = nn.GRU(self.input_dim, self.hidden_dim, self.num_layers, batch_first=True)

        # Define the output layer
        self.linear = LinearZeroInit(self.hidden_dim, output_dim)

        # do_init
        self.do_init = True

    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        self.do_init = True

    def forward(self, input):
        # Forward pass through LSTM layer
        # shape of lstm_out: [batch_size, input_size, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both
        # have shape (batch_size, num_layers, hidden_dim).
        if self.do_init:
            gru_out, self.hidden = self.gru(input)
            self.do_init = False
        else:
            gru_out, self.hidden = self.gru(input, self.hidden)

        # self.hidden = hidden[0].to(input.device), hidden[1].to(input.device)

        # Final layer
        y_pred = self.linear(gru_out)
        return y_pred


class GaussianDiag:
    Log2PI = float(np.log(2 * np.pi))

    @staticmethod
    def likelihood(x):
        """
        lnL = -1/2 * { ln|Var| + ((X - Mu)^T)(Var^-1)(X - Mu) + kln(2*PI) }
              k = 1 (Independent)
              Var = logs ** 2
        """
        return -0.5 * (((x) ** 2) + GaussianDiag.Log2PI)

    @staticmethod
    def logp(x):
        likelihood = GaussianDiag.likelihood(x)
        return thops.sum(likelihood, dim=[1, 2])

    @staticmethod
    def sample(z_shape, eps_std=None, device=None):
        eps_std = eps_std or 1
        eps = torch.normal(mean=torch.zeros(z_shape),
                           std=torch.ones(z_shape) * eps_std)
        eps = eps.to(device)
        return eps


class Split2d(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        print("Split2d num_channels:" + str(num_channels))

        self.num_channels = num_channels
        self.conv = Conv2dZeros(num_channels // 2, num_channels)

    def split2d_prior(self, z):
        h = self.conv(z)
        return thops.split_feature(h, "cross")

    def forward(self, input, cond, logdet=0., reverse=False, eps_std=None):
        if not reverse:
            # print("forward Split2d input:" + str(input.shape))
            z1, z2 = thops.split_feature(input, "split")
            # mean, logs = self.split2d_prior(z1)
            logdet = GaussianDiag.logp(z2) + logdet
            return z1, cond, logdet
        else:
            z1 = input
            # print("reverse Split2d z1.shape:" + str(z1.shape))
            # mean, logs = self.split2d_prior(z1)
            z2_shape = list(z1.shape)
            z2_shape[1] = self.num_channels - z1.shape[1]
            z2 = GaussianDiag.sample(z2_shape, eps_std, device=input.device)
            z = thops.cat_feature(z1, z2)
            return z, cond, logdet


def squeeze2d(input, factor=2):
    assert factor >= 1 and isinstance(factor, int)
    if factor == 1:
        return input
    size = input.shape
    B = size[0]
    C = size[1]
    H = size[2]
    W = size[3]
    assert H % factor == 0, "{}".format((H, W))
    x = input.view(B, C, H // factor, factor, W, 1)
    x = x.permute(0, 1, 3, 5, 2, 4)
    x = x.view(B, C * factor, H // factor, W)
    return x


def unsqueeze2d(input, factor=2):
    assert factor >= 1 and isinstance(factor, int)
    # factor2 = factor ** 2
    if factor == 1:
        return input
    size = input.shape
    B = size[0]
    C = size[1]
    H = size[2]
    W = size[3]
    assert C % (factor) == 0, "{}".format(C)
    x = input.view(B, C // factor, factor, 1, H, W)
    x = x.permute(0, 1, 4, 2, 5, 3)
    x = x.view(B, C // (factor), H * factor, W)
    return x


class SqueezeLayer(nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, input, cond=None, logdet=None, reverse=False):
        if not reverse:
            output = squeeze2d(input, self.factor)
            cond_out = squeeze2d(cond, self.factor)
            return output, cond_out, logdet
        else:
            output = unsqueeze2d(input, self.factor)
            cond_output = unsqueeze2d(cond, self.factor)
            return output, cond_output, logdet

    def squeeze_cond(self, cond):
        cond_out = squeeze2d(cond, self.factor)
        return cond_out
