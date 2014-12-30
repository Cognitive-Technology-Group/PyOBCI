#!/usr/bin/env python2

import time
import random
from csv_collector import CSVCollector

collector = CSVCollector(port='/dev/ttyACM0', fname='motor_data.csv')

num_trials = 7

d = [("up", 1), ("down", 2), ("left", 3), ("right", 4)] * num_trials
random.shuffle(d)

print("get ready...")
time.sleep(2)
collector.start()
print("a bit more...")
time.sleep(2)
print("go!")

for i in range(len(d)):
    x, t = d[i]

    print('\n\n\n' + x)
    collector.tag(t)
    time.sleep(5)
    print('\n\n\nbaseline...')
    collector.tag(0)
    time.sleep(5)


print("done!")
collector.tag(0)
    
collector.stop()
