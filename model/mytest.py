#  Copyright (c) 2024. IPCRC, Lab. Jiangnig Wei
#  All rights reserved

"""
@Project: 2023-GCN-action-recognize-tutorial
@FileName: mytest.py
@Description: 测试
@Author: Wei Jiangning
@version: 1.0.0a1.0
@Date: 2024/4/24 15:34 at PyCharm
"""
from mynetwork import SinPositionEncoding

pe = SinPositionEncoding(8, 10, 12)
pe.forward()
