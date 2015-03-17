import htfe as ht
import pickle as pic
import math
import random

############################## HTFE Init ##############################

cs = ht.ComputeSystem()

cs.create(ht._gpu)

prog = ht.ComputeProgram()

if not prog.loadFromFile("htfe.cl", cs):
    print("Could not load program!")

h = ht.HTFE()

minNote = 21 # Inclusive
maxNote = 109 # Exclusive
numNotes = maxNote - minNote

numNotesRoot = int(math.ceil(math.sqrt(numNotes)))

inputSize = numNotesRoot * numNotesRoot

inputWidth = numNotesRoot
inputHeight = numNotesRoot
minInitWeight = -0.1
maxInitWeight = 0.1

layerDescs = []

l1 = ht.LayerDesc()
l1._width = 64
l1._height = 64

l2 = ht.LayerDesc()
l2._width = 44
l2._height = 44

l3 = ht.LayerDesc()
l3._width = 32
l3._height = 32

l4 = ht.LayerDesc()
l4._width = 22
l4._height = 22

layerDescs.append(l1)
layerDescs.append(l2)
layerDescs.append(l3)
layerDescs.append(l4)

h.createRandom(cs, prog, inputWidth, inputHeight, layerDescs, minInitWeight, maxInitWeight)

############################## Training ##############################

f = open("Piano-midi.de.pickle", "rb")
dataset = pic.load(f)

numSequencesUse = min(4, len(dataset["train"]))

trainIterations = 1

for i in range(0, trainIterations):
    for seq in range(0, numSequencesUse):
        for j in range(0, len(dataset["train"][seq])):
            for k in range(0, numNotes):
                h.setInput(k, 0.0)

            for k in dataset["train"][seq][j]:
                h.setInput(int(k) - minNote, 1.0)

            h.activate(cs)
            h.learn(cs)
            h.stepEnd()

        h.clearMemory(cs)

        print("Training sequence " + str(seq + 1) + " out of " + str(numSequencesUse) + " completed.")

    print("Training iteration " + str(i + 1) + " out of " + str(trainIterations) + " completed.")

############################## Testing Predictions ##############################

errorCount = 0.0
totalCount = 0.0

for seq in range(0, numSequencesUse):
    prediction = []

    for j in range(0, len(dataset["train"][seq])):
        currentInput = []

        for k in range(0, numNotes):
            h.setInput(k, 0.0)
            currentInput.append(0.0)

        for k in dataset["train"][seq][j]:
            h.setInput(int(k) - minNote, 1.0)
            currentInput[int(k) - minNote] = 1.0

        if j > 0:
            # Compare prediction to input
            for k in range(0, numNotes):
                if (prediction[k] > 0.5) != (currentInput[k] > 0.5):
                    errorCount += 1

                totalCount += 1
        
        h.activate(cs)

        h.stepEnd()

        prediction = []

        for k in range(0, numNotes):
            prediction.append(h.getPrediction(k))

    h.clearMemory(cs)

    print("Test sequence " + str(seq + 1) + " out of " + str(numSequencesUse) + " tested.")

print("Error percent: " + str(errorCount / totalCount * 100) + "%")