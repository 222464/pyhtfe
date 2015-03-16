import htfe as ht

cs = ht.ComputeSystem()

cs.create(ht._gpu)

prog = ht.ComputeProgram()

if not prog.loadFromFile("htfe.cl", cs):
    print("Could not load program!")

h = ht.HTFE()

inputWidth = 16
inputHeight = 16
minInitWeight = -0.1
maxInitWeight = 0.1

layerDescs = []

l1 = ht.LayerDesc()
l1._width = 32
l1._height = 32

l2 = ht.LayerDesc()
l2._width = 16
l2._height = 16

layerDescs.append(l1)
layerDescs.append(l2)

h.createRandom(cs, prog, inputWidth, inputHeight, layerDescs, minInitWeight, maxInitWeight)

h.activate(cs)
h.learn(cs)
h.stepEnd()
h.clearMemory(cs)

print("Test complete")