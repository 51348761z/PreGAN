
class Container():
	# IPS = ips requirement
	# RAM = ram requirement in MB
	# Size = container size in MB
	def __init__(self, ID, creationID, creationInterval, IPSModel, RAMModel, DiskModel, Environment, HostID = -1):
		self.id = ID
		self.creationID = creationID
		self.ipsmodel = IPSModel
		self.ipsmodel.allocContainer(self)
		self.sla = self.ipsmodel.SLA
		self.rammodel = RAMModel
		self.rammodel.allocContainer(self)
		self.diskmodel = DiskModel
		self.diskmodel.allocContainer(self)
		self.hostid = HostID
		self.env = Environment
		self.createAt = creationInterval
		self.startAt = self.env.interval
		self.totalExecTime = 0
		self.totalMigrationTime = 0
		self.active = True
		self.destroyAt = -1
		self.lastContainerSize = 0

	def getBaseIPS(self):
		return self.ipsmodel.getIPS()

	def getApparentIPS(self):
		# print("enter getApparentIPS")
		hostBaseIPS = self.getHost().getBaseIPS()
		hostIPSCap = self.getHost().ipsCap
		canUseIPS = (hostIPSCap - hostBaseIPS) / len(self.env.getContainersOfHost(self.hostid))
		return min(self.ipsmodel.getMaxIPS(), self.getBaseIPS() + canUseIPS)

	def getRAM(self):
		rsize, rread, rwrite = self.rammodel.ram()
		self.lastContainerSize = rsize
		return rsize, rread, rwrite

	def getDisk(self):
		return self.diskmodel.disk()

	def getContainerSize(self):
		if self.lastContainerSize == 0: self.getRAM()
		return self.lastContainerSize

	def getHostID(self):
		return self.hostid

	def getHost(self):
		# then enter Simulator.getHostByID
		return self.env.getHostByID(self.hostid)

	def allocate(self, hostID, allocBw):
		# Migrate if allocated to a different host
		# Migration time is sum of network latency 
		# and time to transfer container based on 
		# network bandwidth and container size.
		lastMigrationTime = 0
		# 检查是否需要迁移
		if self.hostid != hostID:
			# 计算迁移所需的时间
			lastMigrationTime += self.getContainerSize() / allocBw
			lastMigrationTime += abs(self.env.hostlist[self.hostid].latency - self.env.hostlist[hostID].latency)
		self.hostid = hostID
		# print(f'lastMigrationTime = {lastMigrationTime}')
		return lastMigrationTime

	def execute(self, lastMigrationTime):
		# Migration time is the time to migrate to new host
		# Thus, execution of task takes place for interval
		# time - migration time with apparent ips
		# 确认容器已经分配了主机
		assert self.hostid != -1
		# 将最后一次迁移所花费的时间 lastMigrationTime 累加到总迁移时间 self.totalMigrationTime
		self.totalMigrationTime += lastMigrationTime
		# print(f'self.env.intervaltime = {self.env.intervaltime}')
		# 计算当前执行周期的剩余执行时间
		execTime = self.env.intervaltime - lastMigrationTime
		apparentIPS = self.getApparentIPS() # 获取容器的实际每秒指令执行数
		# 根据 IPS 模型，计算容器完成剩余指令所需要的执行时间。即 (总指令数 - 已完成的指令数) / 每秒执行的指令数
		requiredExecTime = (self.ipsmodel.totalInstructions - self.ipsmodel.completedInstructions) / apparentIPS if apparentIPS else 0
		# 累加总执行时间
		self.totalExecTime += min(execTime, requiredExecTime)
		# print(f'self.totalExecTime = {self.totalExecTime}')
		# 更新 IPS 模型中的 completedInstructions，即在本周期内容器实际完成的指令数。完成的指令数等于 apparentIPS 乘以本周期的实际执行时间
		self.ipsmodel.completedInstructions += apparentIPS * min(execTime, requiredExecTime)

	def allocateAndExecute(self, hostID, allocBw):
		# print('enter allocateAndExecute')
		self.execute(self.allocate(hostID, allocBw))

	def destroy(self):
		self.destroyAt = self.env.interval
		self.hostid = -1
		self.active = False


