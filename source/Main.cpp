#include <htfe/HTFE.h>

#include <time.h>
#include <iostream>

int main() {
	std::mt19937 generator(time(nullptr));

	sys::ComputeSystem cs;

	cs.create(sys::ComputeSystem::_gpu);

	sys::ComputeProgram program;

	program.loadFromFile("resources/htfe.cl", cs);

	htfe::HTFE test;

	std::vector<htfe::HTFE::LayerDesc> layerDescs(4);

	layerDescs[0]._width = 32;
	layerDescs[0]._height = 32;

	layerDescs[1]._width = 28;
	layerDescs[1]._height = 28;

	layerDescs[2]._width = 22;
	layerDescs[2]._height = 22;

	layerDescs[3]._width = 16;
	layerDescs[3]._height = 16;

	test.createRandom(cs, program, 32, 32, layerDescs, -0.05f, 0.05f, generator);

	test.activate(cs);

	test.learn(cs);

	test.clearMemory(cs);

	return 0;
}