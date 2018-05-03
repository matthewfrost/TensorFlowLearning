import csv
import math
import random
import matplotlib.pyplot as plt

sets = 10000
setLength = 28
lines = []
baseSine = []
inverseSine = []
start= 0
for i in range(0, 28):
    data = math.sin((start * math.pi) / 180)
    baseSine.append(data)
    start += 60
start = 0
for i in range(0, 28):
    data = math.sin((start * math.pi) / -180)
    inverseSine.append(data)
    start += 60

for Set in range(0, sets):
    multiplier = 0
    multiplierStore = []
    start = 0
    wave1 = ""
    for i in range(0, 28):
        multiplierStore.append(multiplier)
        wave1 += str(((multiplier * baseSine[i]) + ((random.uniform(-1, 1) * multiplier) / 5)))
        wave1 += ","
        if multiplier < 255:
            multiplier += 25
    for i in range (0, 28):
        multVal = multiplierStore[i]
        wave1 += str((multVal * inverseSine[i]) + ((random.uniform(-1, 1) * multVal) / 5))
        wave1 +=","
    lines.append(wave1)

for Set in range(0, sets):
    multiplier = 0
    multiplierStore = []
    start = 0
    wave2 = ""
    for i in range(0, 28):
        multiplierStore.append(multiplier)
        wave2 += str((multiplier * baseSine[i]) + ((random.uniform(-1, 1) * multiplier) / 5))
        wave2 += ","
        if multiplier < 255:
            multiplier += 10
    for i in range (0, 28):
        multVal = multiplierStore[i]
        wave2 += str((multVal * inverseSine[i]) + ((random.uniform(-1, 1) * multVal) / 5))
        wave2 +=","
    lines.append(wave2)

for Set in range(0, sets):
    multiplier = 0
    multiplierStore = []
    start = 0
    wave3 = ""
    for i in range(0, 28):
        multiplierStore.append(multiplier)
        wave3 += str((multiplier * baseSine[i]) + ((random.uniform(-1, 1) * multiplier) / 5))
        wave3 +=","
        if multiplier < 255:
            multiplier += 50
    for i in range(0, 28):
        multVal = multiplierStore[i]
        wave3 += str((multVal * inverseSine[i]) + ((random.uniform(-1, 1) * multVal) / 5))
        wave3 +=","
    lines.append(wave3)

for Set in range(0, sets):
    wave3 = ""
    plotData = []
    for i in range(0, 56):
        wave3 += "0"
        wave3 +=","
    lines.append(wave3)

random.shuffle(lines)

splitValue = int(len(lines) * 0.8)
train = lines[:splitValue]
test = lines[splitValue:]

with open("MLTrain.csv", "w") as file:
    file.write("1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56\n")
    for text in train:
        file.write(text)

with open("MLTest.csv", "w") as file:
    file.write("1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56\n")
    for text in test:
        file.write(text)