import htfe as ht

cs = ht.ComputeSystem()

cs.create(ht._gpu)

prog = ht.ComputeProgram()

if not prog.loadFromFile("htfe.cl", cs):
    print("Could not load program!")

h = ht.HTFE()

inputWidth = 4
inputHeight = 4
minInitWeight = -0.1
maxInitWeight = 0.1

layerDescs = []

l1 = ht.LayerDesc()
l1._width = 16
l1._height = 16

l2 = ht.LayerDesc()
l2._width = 12
l2._height = 12

l3 = ht.LayerDesc()
l3._width = 8
l3._height = 8

layerDescs.append(l1)
layerDescs.append(l2)
layerDescs.append(l3)

h.createRandom(cs, prog, inputWidth, inputHeight, layerDescs, minInitWeight, maxInitWeight)

sequence = [
        [ 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1 ],
        [ 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1 ],
        [ 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1 ],
        [ 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1 ],
        [ 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1 ],
        [ 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1 ],
        [ 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1 ],
        [ 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1 ],
        [ 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1 ],
        [ 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1 ]
    ]

for p in range(0, 100):
    for i in range(0, len(sequence)):
        for j in range(0, 16):
            h.setInput(j, sequence[i][j])

        h.activate(cs)
        h.learn(cs)
        h.stepEnd()

#h.clearMemory(cs)

for i in range(0, len(sequence)):
    for j in range(0, 16):
        h.setInput(j, sequence[i][j])

    h.activate(cs)
    
    t = ""

    for j in range(0, 16):
        if h.getPrediction(j) > 0.5:
            t += "1 "
        else:
            t += "0 "

    print(t)

    h.stepEnd()

print("Test complete")