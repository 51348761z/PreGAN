from simulator.host.Host import *
from simulator.container.Container import *

class Simulator():
	# Total power in watt
	# Total Router Bw
	# Interval Time in seconds
	def __init__(self, TotalPower, RouterBw, Scheduler, Recovery, ContainerLimit, IntervalTime, hostinit):
		self.totalpower = TotalPower
		self.totalbw = RouterBw
		self.hostlimit = len(hostinit)
		self.scheduler = Scheduler
		self.scheduler.setEnvironment(self)
		self.recovery = Recovery
		self.recovery.setEnvironment(self)
		self.containerlimit = ContainerLimit
		self.hostlist = []
		self.containerlist = []
		self.intervaltime = IntervalTime
		self.interval = 0
		self.inactiveContainers = []
		self.stats = None
		self.addHostlistInit(hostinit)

	def addHostInit(self, IPS, RAM, Disk, Bw, Latency, Powermodel):
		assert len(self.hostlist) < self.hostlimit
		host = Host(len(self.hostlist), IPS, RAM, Disk, Bw, Latency, Powermodel, self)
		self.hostlist.append(host)

	def addHostlistInit(self, hostList):
		assert len(hostList) == self.hostlimit
		for IPS, RAM, Disk, Bw, Latency, Powermodel in hostList:
			self.addHostInit(IPS, RAM, Disk, Bw, Latency, Powermodel)

	def addContainerInit(self, CreationID, CreationInterval, IPSModel, RAMModel, DiskModel):
		container = Container(len(self.containerlist), CreationID, CreationInterval, IPSModel, RAMModel, DiskModel, self, HostID = -1)
		self.containerlist.append(container)
		return container

	def addContainerListInit(self, containerInfoList):
		'''
			min(len(containerInfoList), self.containerlimit-self.getNumActiveContainers()) 计算能够添加的最大容器数量（即当前还可以部署的数量）。
			deployed 是实际要部署的容器信息列表，包含从 containerInfoList 中选出的前面若干个容器，数量不超过当前可用的容器数量限制。
		'''
		deployed = containerInfoList[:min(len(containerInfoList), self.containerlimit-self.getNumActiveContainers())]
		deployedContainers = []
		for CreationID, CreationInterval, IPSModel, RAMModel, DiskModel in deployed:
			# 这里的 addContainerInit 函数会返回一个 Container 对象，并将其添加到 self.containerlist 中
			dep = self.addContainerInit(CreationID, CreationInterval, IPSModel, RAMModel, DiskModel)
			deployedContainers.append(dep)
		# 为了保证 self.containerlist 的长度始终与 self.containerlimit 保持一致。
		# 如果 self.containerlist 当前长度不足 containerlimit，则用 None 填充剩余的位置
		self.containerlist += [None] * (self.containerlimit - len(self.containerlist))
		# 返回已部署容器的 ID
		return [container.id for container in deployedContainers]

	def addContainer(self, CreationID, CreationInterval, IPSModel, RAMModel, DiskModel):
		for i,c in enumerate(self.containerlist):
			if c == None or not c.active:
				container = Container(i, CreationID, CreationInterval, IPSModel, RAMModel, DiskModel, self, HostID = -1)
				self.containerlist[i] = container
				return container

	"""
		addContainerListInit 的方法，目的是将输入的容器信息列表添加到当前的容器列表中，并且保证不超过容器数量限制。
	"""
	def addContainerList(self, containerInfoList):
		deployed = containerInfoList[:min(len(containerInfoList), self.containerlimit-self.getNumActiveContainers())]
		deployedContainers = []
		for CreationID, CreationInterval, IPSModel, RAMModel, DiskModel in deployed:
			dep = self.addContainer(CreationID, CreationInterval, IPSModel, RAMModel, DiskModel)
			deployedContainers.append(dep)
		return [container.id for container in deployedContainers]

	def getContainersOfHost(self, hostID):
		containers = []
		for container in self.containerlist:
			if container and container.hostid == hostID:
				containers.append(container.id)
		return containers

	def getContainerByID(self, containerID):
		return self.containerlist[containerID]

	def getContainerByCID(self, creationID):
		for c in self.containerlist + self.inactiveContainers:
			if c and c.creationID == creationID:
				return c

	def getHostByID(self, hostID):
		return self.hostlist[hostID]

	"""
		getCreationIDs 方法的作用是根据给定的迁移信息和容器 ID 列表，返回与这些容器对应的 creationID 列表
	"""
	def getCreationIDs(self, migrations, containerIDs):
		creationIDs = []
		for decision in migrations:
			if decision[0] in containerIDs: creationIDs.append(self.containerlist[decision[0]].creationID)
		return creationIDs

	def getPlacementPossible(self, containerID, hostID):
		# print("enter getPlacementPossible")
		container = self.containerlist[containerID]
		host = self.hostlist[hostID]
		ipsreq = container.getBaseIPS()
		ramsizereq, ramreadreq, ramwritereq = container.getRAM()
		disksizereq, diskreadreq, diskwritereq = container.getDisk()
		ipsavailable = host.getIPSAvailable()
		ramsizeav, ramreadav, ramwriteav = host.getRAMAvailable()
		disksizeav, diskreadav, diskwriteav = host.getDiskAvailable()
		return (ipsreq <= ipsavailable and \
				ramsizereq <= ramsizeav and \
				# ramreadreq <= ramreadav and \
				# ramwritereq <= ramwriteav and \
				disksizereq <= disksizeav \
				# diskreadreq <= diskreadav and \
				# diskwritereq <= diskwriteav
				)

	def addContainersInit(self, containerInfoListInit):
		self.interval += 1
		deployed = self.addContainerListInit(containerInfoListInit)
		# 返回已部署容器的 ID
		return deployed

	def allocateInit(self, decision):
		migrations = []
		# 计算每个迁移的带宽分配。self.totalbw 表示总的网络带宽，除以决策中的条目数，计算出每个迁移可以分配的路由器带宽
		routerBwToEach = self.totalbw / len(decision)
		for (cid, hid) in decision:
			container = self.getContainerByID(cid)
			# 确保该容器尚未被分配到任何主机（即 container.getHostID() 应该等于 -1），保证此时的容器没有分配主机资源
			assert container.getHostID() == -1
			# 获取决策中目标主机 hid 的迁移列表，表示有多少个容器要分配到该主机
			numberAllocToHost = len(self.scheduler.getMigrationToHost(hid, decision))
			# 取 主机均分给每个容器带宽 和 路由器带宽 之间的最小值
			allocbw = min(self.getHostByID(hid).bwCap.downlink / numberAllocToHost, routerBwToEach)
			# 检查容器 cid 是否可以被分配到主机 hid。如果返回 True，则允许执行迁移操作。
			if self.getPlacementPossible(cid, hid):
				# 如果容器当前的主机 ID 与目标主机 ID 不同，则将该容器的迁移信息 (cid, hid) 添加到 migrations 列表中。
				if container.getHostID() != hid:
					migrations.append((cid, hid))
				# 将容器分配到目标主机 hid，并分配带宽 allocbw，执行该迁移操作。
				container.allocateAndExecute(hid, allocbw)
			# destroy pointer to this unallocated container as book-keeping is done by workload model
			else: 
				# 表示容器无法分配到目标主机，则将 self.containerlist[cid] 设为 None，表示这个容器被移除
				self.containerlist[cid] = None
		# 返回 migrations 列表，包含所有成功迁移的容器 ID 及其目标主机 ID。
		return migrations

	def destroyCompletedContainers(self):
		destroyed = []
		for i,container in enumerate(self.containerlist):
			# container.getBaseIPS() 返回0, 说明该容器的计算任务已经完成，可以销毁
			if container and container.getBaseIPS() == 0:
				container.destroy()
				self.containerlist[i] = None
				self.inactiveContainers.append(container)
				destroyed.append(container)
		return destroyed

	def getNumActiveContainers(self):
		num = 0 
		for container in self.containerlist:
			if container and container.active: num += 1
		return num

	def getSelectableContainers(self):
		selectable = []
		for container in self.containerlist:
			if container and container.active and container.getHostID() != -1:
				selectable.append(container.id)
		return selectable

	def addContainers(self, newContainerList):
		self.interval += 1
		destroyed = self.destroyCompletedContainers()
		deployed = self.addContainerList(newContainerList)
		return deployed, destroyed

	"""
		该方法返回一个列表，列表中的每个元素代表 containerlist 中每个容器的主机分配状态。
		如果容器是活动的，则返回该容器当前所在的主机 ID；
		否则返回 -1，表示该容器未被分配或不活跃。
	"""
	def getActiveContainerList(self):
		return [c.getHostID() if c and c.active else -1 for c in self.containerlist]

	"""
		该方法返回一个列表，每个元素代表相应主机上运行的容器数量。
	"""
	def getContainersInHosts(self):
		return [len(self.getContainersOfHost(host)) for host in range(self.hostlimit)]

	"""
		simulationStep 函数函数根据给定的调度决策调整容器分配。
		它计算带宽，执行容器迁移，处理未分配的容器，并记录迁移的容器列表
	"""
	def simulationStep(self, decision):
		routerBwToEach = self.totalbw / len(decision) if len(decision) > 0 else self.totalbw
		migrations = [] # 用于存储需要迁移的容器和目标主机
		containerIDsAllocated = [] # 记录本次步骤中成功分配的容器 ID
		for (cid, hid) in decision:
			container = self.getContainerByID(cid)
			currentHostID = self.getContainerByID(cid).getHostID()
			currentHost = self.getHostByID(currentHostID)
			targetHost = self.getHostByID(hid)
			migrateFromNum = len(self.scheduler.getMigrationFromHost(currentHostID, decision))
			migrateToNum = len(self.scheduler.getMigrationToHost(hid, decision))
			allocbw = min(targetHost.bwCap.downlink / migrateToNum, currentHost.bwCap.uplink / migrateFromNum, routerBwToEach)
			# 迁移和分配容器
			if hid != self.containerlist[cid].hostid and self.getPlacementPossible(cid, hid):
				migrations.append((cid, hid))
				container.allocateAndExecute(hid, allocbw)
				containerIDsAllocated.append(cid)
		# destroy pointer to unallocated containers as book-keeping is done by workload model
		for (cid, hid) in decision:
			if self.containerlist[cid].hostid == -1: self.containerlist[cid] = None
		# 处理未分配的容器
		for i,container in enumerate(self.containerlist):
			if container and i not in containerIDsAllocated:
				container.execute(0) # 传入 0 表示这些容器没有经历迁移时间
		return migrations # 返回需要迁移的容器和目标主机的对列表
