import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from .constants import *
from .dlutils import *

## FPE
class FPE_16(nn.Module):
	def __init__(self):
		super(FPE_16, self).__init__()
		self.name = 'FPE_16'
		self.lr = 0.0001
		self.n_hosts = 16
		self.n_feats = 3 * self.n_hosts
		self.n_window = 3 # w_size = 5
		self.n_latent = 10
		self.n_hidden = 16
		self.n = self.n_window * self.n_feats + self.n_hosts * self.n_hosts
		self.gru = nn.GRU(self.n_window, self.n_window, 1) # (3, 3, 1)
		"""
			dgl.graph((src_ids, dst_ids)) 使用了 DGL（Deep Graph Library），用于定义图结构，
			src_ids 和 dst_ids 表示图中节点的连接关系（边的源节点和目标节点）
		"""
		src_ids = torch.tensor(list(range(self.n_feats))); dst_ids = torch.tensor([self.n_feats] * self.n_feats)
		self.gat = GAT(dgl.graph((src_ids, dst_ids)), self.n_window, self.n_window)
		self.mha = nn.MultiheadAttention(self.n_feats * 2 + 1, 1)

		self.encoder = nn.Sequential(
			# nn.LeakyReLU() 参数 True 表示执行 in-place 操作，即直接在原地修改输入数据而不占用额外的内存。
			nn.Linear(self.n_window * (self.n_feats * 2 + 1), self.n_hosts * self.n_latent), nn.LeakyReLU(True),
		)
		"""
			self.anomaly_decoder：一个全连接网络（线性层），将潜在特征 n_latent 映射到 2 维输出，
			后接 Softmax 激活函数，用于将输出转化为分类概率（例如，用于判断异常或正常的概率）。	
		"""
		self.anomaly_decoder = nn.Sequential(
			nn.Linear(self.n_latent, 2), nn.Softmax(dim=0),
		)
		"""
			self.prototype_decoder：将潜在特征 n_latent 映射到维度为 PROTO_DIM 的原型特征，
			后接 Sigmoid 激活函数，使输出落在 0 到 1 之间（通常用来表示概率或归一化的原型特征）。
		"""
		self.prototype_decoder = nn.Sequential(
			# PROTO_DIM = 2
			nn.Linear(self.n_latent, PROTO_DIM), nn.Sigmoid(),
		)
		self.prototype = [torch.rand(PROTO_DIM, requires_grad=False, dtype=torch.double) for _ in range(3)]

	"""
		encode 函数的目的是通过一系列神经网络组件对输入的时间序列数据 t 和调度决策 s 进行编码，最后生成一个潜在的表征 t。
	"""
	def encode(self, t, s):
		h = torch.randn(1, self.n_window, dtype=torch.double) # h shape (1, 3)
		gru_t, _ = self.gru(torch.t(t), h) # 取T转秩 = (48, 3)
		gru_t = torch.t(gru_t) # gru_t shape: (3, 48)
		graph = torch.cat((t, torch.zeros(self.n_window, 1)), dim=1) # graph shape: (3, 49)
		gat_t = self.gat(torch.t(graph)) # gat_t shape: (49, 3)
		gat_t = torch.t(gat_t) # gat_t transposes into (3, 49)
		concat_t = torch.cat((gru_t, gat_t), dim=1) # concat_t shape: (3, 48+49)
		o, _ = self.mha(concat_t, concat_t, concat_t) # 多头注意力层不会改变形状
		# 将注意力机制的输出 o 展平成一维向量，然后通过编码器 self.encoder()，生成潜在表征。
		t = self.encoder(o.view(-1)).view(self.n_hosts, self.n_latent)	
		# 编码器的输出重新调整形状为 (self.n_hosts, self.n_latent)，这里 n_hosts 表示主机数量，n_latent 表示潜在维度。
		return t

	"""
		该方法遍历输入 t 中的每个元素 elem，使用 anomaly_decoder 进行解码，
		得到异常检测的得分（概率），并将其 reshape 成行向量后添加到 anomaly_scores 列表中。
		得到16个二元组D，对应每个主机 host 的异常检测结果。D[1] >= D[0] 表示 host 异常。(论文P4-(6))
	"""
	def anomaly_decode(self, t):
		anomaly_scores = []
		for elem in t:
			anomaly_scores.append(self.anomaly_decoder(elem).view(1, -1))	
		return anomaly_scores

	"""
		输入 t 是潜在特征表示的张量。
		该方法遍历输入 t 中的每个元素 elem，使用 prototype_decoder 进行解码，
		得到对应的原型 Embedding 向量，并将其添加到 prototypes 列表中。
		最终返回所有的原型向量。 若主机i检测到异常，则其异常原型为 prototypes[i]，反之，prototypes[i] = 0。(论文P4-(6))
	"""
	def prototype_decode(self, t):
		prototypes = []
		for elem in t:
			prototypes.append(self.prototype_decoder(elem))	
		return prototypes

	def forward(self, t, s):
		t = self.encode(t, s) # t shape: (16, 10) - (self.hosts, self.latent)
		anomaly_scores = self.anomaly_decode(t)
		prototypes = self.prototype_decode(t)
		return anomaly_scores, prototypes

# Generator Network : Input = Embedding, Schedule; Output = New Schedule
class Gen_16(nn.Module):
	def __init__(self):
		super(Gen_16, self).__init__()
		self.name = 'Gen_16'
		self.lr = 0.00005
		self.n_hosts = 16
		self.n_hidden = 64
		self.n = self.n_hosts * PROTO_DIM + self.n_hosts * self.n_hosts
		self.delta = nn.Sequential(
			nn.Linear(self.n, self.n_hidden),
			nn.LeakyReLU(True),
			nn.Linear(self.n_hidden, self.n_hosts * self.n_hosts),
			nn.Tanh(), # 输出层的激活函数 (Tanh，用于将输出限制在 -1 到 1)
		)

	def forward(self, e, s):
		# (torch.cat((e.view(-1), s.view(-1)))) shape: (16*2 + 16*16)
		del_s = 4 * self.delta(torch.cat((e.view(-1), s.view(-1)))) # del_s shape: (16*16)
		"""
			将状态变化量 del_s 添加到原始决策 s 上，得到更新后的状态。
			s = 16*16 one-hot matrix represents (containerID, toHostID)
		"""
		return s + del_s.reshape(self.n_hosts, self.n_hosts)

# Discriminator Network : Input = Schedule, New Schedule; Output = Likelihood scores
class Disc_16(nn.Module):
	def __init__(self):
		super(Disc_16, self).__init__()
		self.name = 'Disc_16'
		self.lr = 0.00005
		self.n_hosts = 16
		self.n_hidden = 64
		self.n = self.n_hosts * self.n_hosts + self.n_hosts * self.n_hosts
		self.probs = nn.Sequential(
			nn.Linear(self.n, self.n_hidden), nn.LeakyReLU(True),
			nn.Linear(self.n_hidden, 2), nn.Softmax(dim=0),
		)

	def forward(self, o, n):
		#(torch.cat((o.view(-1), n.view(-1)))) shape: (16*16 + 16*16) 
		probs = self.probs(torch.cat((o.view(-1), n.view(-1))))
		# probs[1] >= probs[0], 取新决策 论文P4-(9)
		return probs


## FPE
class FPE_50(nn.Module):
	def __init__(self):
		super(FPE_50, self).__init__()
		self.name = 'FPE_50'
		self.lr = 0.0001
		self.n_hosts = 50
		self.n_feats = 3 * self.n_hosts
		self.n_window = 3 # w_size = 5
		self.n_latent = 10
		self.n_hidden = 50
		self.n = self.n_window * self.n_feats + self.n_hosts * self.n_hosts
		self.gru = nn.GRU(self.n_window, self.n_window, 1)
		src_ids = torch.tensor(list(range(self.n_feats))); dst_ids = torch.tensor([self.n_feats] * self.n_feats)
		self.gat = GAT(dgl.graph((src_ids, dst_ids)), self.n_window, self.n_window)
		self.mha = nn.MultiheadAttention(self.n_feats * 2 + 1, 1)
		self.encoder = nn.Sequential(
			nn.Linear(self.n_window * (self.n_feats * 2 + 1), self.n_hosts * self.n_latent), nn.LeakyReLU(True),
		)
		self.anomaly_decoder = nn.Sequential(
			nn.Linear(self.n_latent, 2), nn.Softmax(dim=0),
		)
		self.prototype_decoder = nn.Sequential(
			nn.Linear(self.n_latent, PROTO_DIM), nn.Sigmoid(),
		)
		self.prototype = [torch.rand(PROTO_DIM, requires_grad=False, dtype=torch.double) for _ in range(3)]

	def encode(self, t, s):
		h = torch.randn(1, self.n_window, dtype=torch.double)
		gru_t, _ = self.gru(torch.t(t), h)
		gru_t = torch.t(gru_t)
		graph = torch.cat((t, torch.zeros(self.n_window, 1)), dim=1)
		gat_t = self.gat(torch.t(graph))
		gat_t = torch.t(gat_t)
		concat_t = torch.cat((gru_t, gat_t), dim=1)
		o, _ = self.mha(concat_t, concat_t, concat_t)
		t = self.encoder(o.view(-1)).view(self.n_hosts, self.n_latent)	
		return t

	def anomaly_decode(self, t):
		anomaly_scores = []
		for elem in t:
			anomaly_scores.append(self.anomaly_decoder(elem).view(1, -1))	
		return anomaly_scores

	def prototype_decode(self, t):
		prototypes = []
		for elem in t:
			prototypes.append(self.prototype_decoder(elem))	
		return prototypes

	def forward(self, t, s):
		t = self.encode(t, s)
		anomaly_scores = self.anomaly_decode(t)
		prototypes = self.prototype_decode(t)
		return anomaly_scores, prototypes

## Simple Multi-Head Self-Attention Model
class Attention_50(nn.Module):
	def __init__(self):
		super(Attention_50, self).__init__()
		self.name = 'Attention_50'
		self.lr = 0.0008
		self.n_hosts = 50
		self.n_feats = 3 * self.n_hosts
		self.n_window = 3 # w_size = 5
		self.n_latent = 10
		self.n_hidden = 16
		self.n = self.n_window * self.n_feats + self.n_hosts * self.n_hosts
		# self.atts = [ nn.Sequential( nn.Linear(self.n, self.n_feats * self.n_feats), 
		# 		nn.Sigmoid())	for i in range(1)]
		# self.encoder_atts = nn.ModuleList(self.atts)
		self.encoder = nn.Sequential(
			nn.Linear(self.n_window * self.n_feats, self.n_hosts * self.n_latent), nn.LeakyReLU(True),
		)
		self.anomaly_decoder = nn.Sequential(
			nn.Linear(self.n_latent, 2), nn.Softmax(dim=0),
		)
		self.prototype_decoder = nn.Sequential(
			nn.Linear(self.n_latent, PROTO_DIM), nn.Sigmoid(),
		)
		self.prototype = [torch.rand(PROTO_DIM, requires_grad=False, dtype=torch.double) for _ in range(3)]

	def encode(self, t, s):
		# for at in self.encoder_atts:
		# 	inp = torch.cat((t.view(-1), s.view(-1)))
		# 	ats = at(inp).reshape(self.n_feats, self.n_feats)
		# 	t = torch.matmul(t, ats)	
		t = self.encoder(t.view(-1)).view(self.n_hosts, self.n_latent)	
		return t

	def anomaly_decode(self, t):
		anomaly_scores = []
		for elem in t:
			anomaly_scores.append(self.anomaly_decoder(elem).view(1, -1))	
		return anomaly_scores

	def prototype_decode(self, t):
		prototypes = []
		for elem in t:
			prototypes.append(self.prototype_decoder(elem))	
		return prototypes

	def forward(self, t, s):
		t = self.encode(t, s)
		anomaly_scores = self.anomaly_decode(t)
		prototypes = self.prototype_decode(t)
		return anomaly_scores, prototypes

# Generator Network : Input = Schedule, Embedding; Output = New Schedule
class Gen_50(nn.Module):
	def __init__(self):
		super(Gen_50, self).__init__()
		self.name = 'Gen_50'
		self.lr = 0.00003
		self.n_hosts = 50
		self.n_hidden = 64
		self.n = self.n_hosts * PROTO_DIM + self.n_hosts * self.n_hosts
		self.delta = nn.Sequential(
			nn.Linear(self.n, self.n_hidden), nn.LeakyReLU(True),
			nn.Linear(self.n_hidden, self.n_hosts * self.n_hosts), nn.Tanh(),
		)

	def forward(self, e, s):
		del_s = 4 * self.delta(torch.cat((e.view(-1), s.view(-1))))
		return s + del_s.reshape(self.n_hosts, self.n_hosts)

# Discriminator Network : Input = Schedule, New Schedule; Output = Likelihood scores
class Disc_50(nn.Module):
	def __init__(self):
		super(Disc_50, self).__init__()
		self.name = 'Disc_50'
		self.lr = 0.00003
		self.n_hosts = 50
		self.n_hidden = 64
		self.n = self.n_hosts * self.n_hosts + self.n_hosts * self.n_hosts
		self.probs = nn.Sequential(
			nn.Linear(self.n, self.n_hidden), nn.LeakyReLU(True),
			nn.Linear(self.n_hidden, 2), nn.Softmax(dim=0),
		)

	def forward(self, o, n):
		probs = self.probs(torch.cat((o.view(-1), n.view(-1))))
		return probs


############## PreGANPlus Models ##############

# Transformer Model
class Transformer_16(nn.Module):
	def __init__(self):
		super(Transformer_16, self).__init__()
		self.name = 'Transformer_16'
		self.lr = 0.0001
		self.n_hosts = 16
		feats = 3 * self.n_hosts
		self.n_feats = 3 * self.n_hosts
		self.n_window = 3 # w_size = 5
		self.n_latent = 10
		self.n_hidden = 16
		self.n = self.n_window * self.n_feats + self.n_hosts * self.n_hosts
		src_ids = torch.tensor(list(range(self.n_feats))); dst_ids = torch.tensor([self.n_feats] * self.n_feats)
		self.gat = GAT(dgl.graph((src_ids, dst_ids)), self.n_window, self.n_window)
		self.time_encoder = nn.Sequential(
			nn.Linear(feats, feats * 2 + 1), 
		)
		self.pos_encoder = PositionalEncoding(feats * 2 + 1, 0.1, self.n_window)
		encoder_layers = TransformerEncoderLayer(d_model=feats * 2 + 1, nhead=1, dropout=0.1)
		self.encoder = TransformerEncoder(encoder_layers, 1)
		a_decoder_layers = TransformerDecoderLayer(d_model=feats * 2 + 1, nhead=1, dropout=0.1)
		self.anomaly_decoder = TransformerDecoder(a_decoder_layers, 1)
		self.anomaly_decoder2 = nn.Sequential(
			nn.Linear((feats * 2 + 1) * self.n_window * self.n_window, 2 * self.n_hosts), 
		)
		self.softm = nn.Softmax(dim=1)
		p_decoder_layers = TransformerDecoderLayer(d_model=feats * 2 + 1, nhead=1, dropout=0.1)
		self.prototype_decoder = TransformerDecoder(p_decoder_layers, 1)
		self.prototype_decoder2 = nn.Sequential(
			nn.Linear((feats * 2 + 1) * self.n_window * self.n_window, PROTO_DIM * self.n_hosts), 
		)
		self.prototype = [torch.rand(PROTO_DIM, requires_grad=False, dtype=torch.double) for _ in range(3)]

	def encode(self, t, s):
		t = torch.squeeze(t, 1)
		graph = torch.cat((t, torch.zeros(self.n_window, 1)), dim=1)
		gat_t = self.gat(torch.t(graph))
		gat_t = torch.t(gat_t)
		o = torch.cat((t, gat_t), dim=1)
		t = o * math.sqrt(self.n_feats)
		t = self.pos_encoder(t) # window size, batch size (1), feats (3 metrics * 16 hosts)
		memory = self.encoder(t)	
		return memory

	def anomaly_decode(self, t, memory):
		anomaly_scores = self.anomaly_decoder(t, memory)
		anomaly_scores = self.anomaly_decoder2(anomaly_scores.view(-1)).view(-1, 1, 2)
		return anomaly_scores

	def prototype_decode(self, t, memory):
		prototypes = self.prototype_decoder(t, memory)
		prototypes = self.prototype_decoder2(prototypes.view(-1)).view(-1, PROTO_DIM)
		return prototypes

	def forward(self, t, s):
		encoded_t = self.time_encoder(t).unsqueeze(dim=1).expand(-1, self.n_window, -1)
		t = t.unsqueeze(dim=1)
		memory = self.encode(t, s)
		anomaly_scores = self.anomaly_decode(encoded_t, memory)
		prototypes = self.prototype_decode(encoded_t, memory)
		return anomaly_scores, prototypes
