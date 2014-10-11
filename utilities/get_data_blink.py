#!/usr/bin/env python2

import time
import random
from csv_collector import CSVCollector

collector = CSVCollector(port='/dev/ttyACM1', fname='eye_blink.csv')

#d = [("left hand", -1), ("right hand", 1)]

print("get ready...")
time.sleep(2)
print("Go!")
collector.start()

time.sleep(2)
print("really starting")
# for i in range(5):
#     random.shuffle(d)
#     for x, t in d:
#         print(x)
#         collector.tag(t)
#         time.sleep(5)
#         print('baseline...')
#         collector.tag(0)
#         time.sleep(5)

time.sleep(5)

print("done!")
collector.tag(0)
    
collector.stop()
