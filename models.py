import torch
from transformers import AutoTokenizer, AutoModel, pipeline
import torch.nn.functional as F
import numpy as np
import torch.nn as nn


tokenizer = AutoTokenizer.from_pretrained("DNABert-Prunning", do_lower_case=False )
model = AutoModel.from_pretrained("DNABert-Prunning")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class MyModel(nn.Module):

    def __init__(self, channels=256, r=4):
        super(MyModel, self).__init__()
        #         self.linear = nn.Linear(64*41, 1)
        inter_channels = int(channels // r)
        self.local_att = nn.Sequential(
            nn.Conv1d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            #             nn.LayerNorm(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            #             nn.LayerNorm(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            #             nn.LayerNorm(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            #             nn.LayerNorm(channels),
        )
        self.layer_norm = nn.LayerNorm(256)
        # 加载bert模型
        self.bert = AutoModel.from_pretrained("/dnabert3/", output_hidden_states=True)
        self.num_hidden_layers = 2
        self.hidden_size = 768
        self.hiddendim_fc = 256
        self.dropout = nn.Dropout(0.1)

        q_t = np.random.normal(loc=0.0, scale=0.1, size=(1, self.hidden_size))
        self.q = nn.Parameter(torch.from_numpy(q_t)).float()
        w_ht = np.random.normal(loc=0.0, scale=0.1, size=(self.hidden_size, self.hiddendim_fc))
        self.w_h = nn.Parameter(torch.from_numpy(w_ht)).float()
        #          seq——feature特征编码
        self.src_emb = nn.Embedding(4800, 16)
        #         self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(41, 16), freeze=True)
        #         self.layer_norm = nn.LayerNorm(16)
        #         encoder_layer = nn.TransformerEncoderLayer(d_model=16, nhead=2, dim_feedforward=32)
        #         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.launch_seq_lstm = nn.LSTM(16, 16, bidirectional=True, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(32 * 36, 256)
        # 最后的预测层
        self.predictor = nn.Sequential(
            #             nn.Linear(31488, 256),
            #             nn.ReLU(),
            #             nn.Linear(256,1),
            nn.Sigmoid()
        )
        self.linear_1 = nn.Sequential(
            nn.Linear(256, 1),
            #             nn.ReLU(),
            #             nn.Sigmoid()
        )

    def forward(self, src, launch_seq):
        """
        :param src: 分词后的推文数据
        """
        output = self.bert(**src)
        all_hidden_states = torch.stack(output[2])
        hidden_states = torch.stack([all_hidden_states[layer_i][:, 0].squeeze()
                                     for layer_i in range(1, self.num_hidden_layers + 1)], dim=-1)
        hidden_states = hidden_states.view(-1, self.num_hidden_layers, self.hidden_size)
        out = self.attention(hidden_states)
        out = self.layer_norm(out)
        output = self.dropout(out)

        launch_conv = self.src_emb(launch_seq)  # [batch_size, seq_len, d_model]
        output_launch, (_, _) = self.launch_seq_lstm(launch_conv)
        lauch_layer = output_launch.contiguous().view(launch_seq.shape[0], -1)
        hidden = self.linear(lauch_layer)
        output_final = output + hidden
        #         print(output_final,"ef")
        output_final = self.layer_norm(output_final)
        xl = self.local_att(output_final.unsqueeze(-1))
        #         print(xl.shape,"wq")
        xg = self.global_att(output_final.unsqueeze(-1))
        xlg = xl + xg
        #         print(xlg.shape,"fj")
        wei = self.predictor(xlg).squeeze()
        xo = 2 * output * wei + 2 * hidden * (1 - wei)
        #         print(xo.shape,"qq")
        xo = xo.view(xo.size(0), -1)
        #         xo = self.layer_norm(xo)
        #         print(xo.shape,"ii")
        xo = self.linear_1(xo)
        return self.predictor(xo)

    def attention(self, h):
        v = torch.matmul(self.q.to(device), h.transpose(-2, -1)).squeeze(1)
        #         v = v.to(device)
        v = F.softmax(v, -1)
        v_temp = torch.matmul(v.unsqueeze(1), h).transpose(-2, -1)
        v = torch.matmul(self.w_h.transpose(1, 0).to(device), v_temp).squeeze(2)
        return v.to(device)
    
# 由于inputs是字典类型的，定义一个辅助函数帮助to(device)
def to_device(dict_tensors):
    result_tensors = {}
    for key, value in dict_tensors.items():
        result_tensors[key] = value.to(device)
    return result_tensors


class AttentionPooling(nn.Module):
    def __init__(self, num_layers, hidden_size, hiddendim_fc):
        super(AttentionPooling, self).__init__()
        self.num_hidden_layers = 2
        self.hidden_size = 768
        self.hiddendim_fc = hiddendim_fc
        self.dropout = nn.Dropout(0.1)

        q_t = np.random.normal(loc=0.0, scale=0.1, size=(1, self.hidden_size))
        self.q = nn.Parameter(torch.from_numpy(q_t)).float()
        w_ht = np.random.normal(loc=0.0, scale=0.1, size=(self.hidden_size, self.hiddendim_fc))
        self.w_h = nn.Parameter(torch.from_numpy(w_ht)).float()

    def forward(self, all_hidden_states):
        hidden_states = torch.stack([all_hidden_states[layer_i][:, 0].squeeze()
                                     for layer_i in range(1, self.num_hidden_layers+1)], dim=-1)
        hidden_states = hidden_states.view(-1, self.num_hidden_layers, self.hidden_size)
        out = self.attention(hidden_states)
        out = self.dropout(out)
        return out

    def attention(self, h):
        v = torch.matmul(self.q, h.transpose(-2, -1)).squeeze(1)
        v = F.softmax(v, -1)
        v_temp = torch.matmul(v.unsqueeze(1), h).transpose(-2, -1)
        v = torch.matmul(self.w_h.transpose(1, 0), v_temp).squeeze(2)
        return v
