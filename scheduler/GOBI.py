import sys
sys.path.append('scheduler/BaGTI/')

from .Scheduler import *
from .BaGTI.train import *
from copy import deepcopy

class GOBIScheduler(Scheduler):
	def __init__(self, data_type):
		super().__init__()
		# in here data type is energy_latency_16 by default
		self.model = eval(data_type+"()") # self.model = energy_latency_16()
		self.model, _, _, _ = load_model(data_type, self.model, data_type)
		self.data_type = data_type
		self.hosts = int(data_type.split('_')[-1])
		self.result_cache = None
		dtl = data_type.split('_')
		_, _, self.max_container_ips = eval("load_"+'_'.join(dtl[:-1])+"_data("+dtl[-1]+")")

	'''
		run_GOBI 方法的主要功能：
		1.获取每个主机和容器的 CPU 使用情况。
		2.根据某种策略进行容器分配（独热编码表示）。
		3.调用优化函数 opt 进行优化计算。
		4.返回基于优化结果的容器部署决策。
	'''
	def run_GOBI(self):
		# cpu 列表保存了所有主机的 CPU 使用情况，getCPU() 返回主机的 CPU 使用百分比，除以 100 将其归一化。
		cpu = [host.getCPU()/100 for host in self.env.hostlist]
		cpu = np.array([cpu]).transpose() # zeros; cpu shape is (16, 1) after transpose
		if 'latency' in self.model.name: # here self.model.name = energy_latency_16, 针对延迟模型
			# getApparentIPS() 返回容器的 "Apparent Instructions Per Second" (IPS)，这是容器的计算需求，
			# 除以 self.max_container_ips 归一化后，表示容器的相对负载。
			cpuC = [(c.getApparentIPS()/self.max_container_ips if c else 0) for c in self.env.containerlist] # enter Container.getApparentIPS
			# print(f'cpuC before transpose = {cpuC}')
			cpuC = np.array([cpuC]).transpose() # cpuC shape is (16, 1)
			# 这些容器的 CPU 使用情况转置并与主机的 CPU 使用情况拼接，形成一个新的矩阵 cpu。
			cpu = np.concatenate((cpu, cpuC), axis=1) # axis=1, 按列拼接, shape is (16, 2)

		'''
			初始化 alloc 列表，用于存储每个容器的主机分配信息，使用独热编码表示。
			prev_alloc 字典用于保存每个容器的 ID 及其当前的主机分配情况。
		'''
		alloc = []; prev_alloc = {}
		'''
			对每个容器 c：
				如果容器存在且已经分配了主机（getHostID() 不为 -1），则在 oneHot 中设置对应主机的位置为 1。
				如果容器尚未分配主机，则随机选择一个主机进行分配。随机分配后的容器getHostID()仍是-1, 还未真正分配
		'''
		for c in self.env.containerlist:
			oneHot = [0] * len(self.env.hostlist)
			if c: prev_alloc[c.id] = c.getHostID() # if c, prev_alloc[containerID] = ContainerHostID
			if c and c.getHostID() != -1: oneHot[c.getHostID()] = 1
			else: oneHot[np.random.randint(0,len(self.env.hostlist))] = 1
			# 将生成的 oneHot 添加到 alloc 列表中。
			# alloc 的行是容器 ID，列表示主机host ID
			alloc.append(oneHot)
		# print(f'cpu: \n{cpu}')
		# print(f'onhot = \n{oneHot}')
		# print(f'alloc = \n{alloc}')
		# print(f'len alloc = {len(alloc)}')
		init = np.concatenate((cpu, alloc), axis=1)
		init = torch.tensor(init, dtype=torch.float, requires_grad=True)
		"""
			将主机和容器的 CPU 使用情况与独热编码表示的分配情况拼接，形成初始的 init 矩阵。
			init.shape = (16, 18)
			each row represents: (cpu, cpuC, alloc1, alloc2,..., alloc16)
			alloc matrix is one-hot encoding (16*16)
		"""
		'''
			调用之前定义的 opt 函数，传入 init（初始分配信息）、self.model（优化模型）以及其他参数，执行优化。
			result 是优化后的结果，iteration 是迭代次数，fitness 可能是优化后的损失值或模型的适应度值。
		'''
		result, iteration, fitness = opt(init, self.model, [], self.data_type) # opt.py

		# 将 result 中独热编码分配信息部分保存到 self.result_cache 中。
		# 这里提取的是最后 self.hosts 列的数据（对应每个容器的主机分配状态），并将其转换为 NumPy 数组。
		self.result_cache = result[:, -self.hosts:].numpy()

		"""
			遍历之前记录的 prev_alloc 中的容器：
			从优化后的 result 中提取该容器的分配信息（独热编码）。
			根据最大值的位置，确定容器的新主机 new_host。
			如果新主机与之前的主机不同，将容器 ID 和新主机添加到 decision 列表中。
		"""
		decision = []
		for cid in prev_alloc:
			one_hot = result[cid, -self.hosts:].tolist()
			new_host = one_hot.index(max(one_hot))
			"""
				如果容器的先前主机 ID (prev_alloc[cid]) 与新的主机 ID (new_host) 不同，
				则将 (cid, new_host) 添加到 decision 列表中，表示需要将该容器迁移到新的主机
			"""
			if prev_alloc[cid] != new_host: decision.append((cid, new_host))
		return decision

	def selection(self):
		# print('enter selection')
		return []

	def placement(self, containerIDs):
		"""
			np.all(np.array)   对矩阵所有元素做与操作，所有为True则返回True
			np.any(np.array)   对矩阵所有元素做或运算，存在True则返回True
		"""
		first_alloc = np.all([not (c and c.getHostID() != -1) for c in self.env.containerlist])
		decision = self.run_GOBI()
		return decision
