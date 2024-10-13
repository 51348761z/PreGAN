from .IPSM import *

class IPSMBitbrain(IPSM):
	def __init__(self, ips_list, max_ips, duration, SLA):
		super().__init__()
		self.ips_list = ips_list
		self.max_ips = max_ips
		self.SLA = SLA
		self.duration = duration
		self.completedInstructions = 0
		self.totalInstructions = 0

	"""
		getIPS 方法的功能是计算容器的指令执行速率 (IPS, Instructions Per Second) 或根据当前进度返回指令执行量。
		它根据容器的已完成任务量和总任务量，动态返回当前时间段的指令执行速率。
	"""
	def getIPS(self):
		# print('enter getIPS')
		# 如果 self.totalInstructions == 0，表示还未计算该容器的总任务量，因此需要计算。
		if self.totalInstructions == 0:
			"""
				遍历 self.ips_list 中从头到 duration 的部分，
				将每个 ips 乘以容器环境中的时间间隔 self.container.env.intervaltime，
				计算并累加总的指令数 self.totalInstructions。
			"""
			for ips in self.ips_list[:self.duration]: self.totalInstructions += ips * self.container.env.intervaltime
		# 如果 self.completedInstructions 小于 self.totalInstructions，表示任务尚未完成，应该继续返回当前的 IPS。
		if self.completedInstructions < self.totalInstructions:
			return self.ips_list[(self.container.env.interval - self.container.startAt) % len(self.ips_list)]
		# 如果 self.completedInstructions 大于或等于 self.totalInstructions，则任务已经完成，
		# 没有更多的指令需要执行，因此返回 0 表示没有剩余的指令执行。
		return 0

	def getMaxIPS(self):
		return self.max_ips
