import os
import logging
import json
import re
from subprocess import call
from .ColorUtils import *

"""
	printDecisionAndMigrations 方法的作用是打印出容器调度决策 (decision) 和迁移操作 (migrations)，
	并且通过颜色标识哪些决策没有导致容器迁移。
"""
def printDecisionAndMigrations(decision, migrations):
	print('Decision: [', end='')
	for i, d in enumerate(decision):
		if d not in migrations: print(color.FAIL, end='')
		print(d, end='')
		if d not in migrations: print(color.ENDC, end='')
		print(',', end='') if i != len(decision)-1 else print(']')
	print()


def unixify(paths):
	for path in paths:
		for file in os.listdir(path):
			if '.py' in file or '.sh' in file:
				_ = os.system("bash -c \"dos2unix "+path+file+" 2&> /dev/null\"")
