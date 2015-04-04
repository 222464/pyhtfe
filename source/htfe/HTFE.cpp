#include "HTFE.h"

#include <iostream>
#include <time.h>

using namespace htfe;

struct Uint2 {
	unsigned int _x, _y;
};

struct Float2 {
	float _x, _y;
};

struct Float4 {
	float _x, _y, _z, _w;
};

struct Int2 {
	int _x, _y;
};

void HTFE::createRandom(sys::ComputeSystem &cs, sys::ComputeProgram &program, int inputWidth, int inputHeight, const std::vector<LayerDesc> &layerDescs, float minInitWeight, float maxInitWeight) {
	std::mt19937 generator(time(nullptr));

	std::uniform_int_distribution<int> seedDist(0, 99999);

	_inputWidth = inputWidth;
	_inputHeight = inputHeight;

	_layerDescs = layerDescs;

	_layers.resize(_layerDescs.size());

	cl::Kernel initializeLayerHiddenSpatialKernel = cl::Kernel(program.getProgram(), "initializeLayerHiddenSpatial");
	cl::Kernel initializeLayerHiddenTemporalKernel = cl::Kernel(program.getProgram(), "initializeLayerHiddenTemporal");

	_input.clear();
	_input.resize(_inputWidth * _inputHeight, 0.0f);

	_prediction.clear();
	_prediction.resize(_inputWidth * _inputHeight, 0.0f);

	_inputImage = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _inputWidth, _inputHeight);
	_inputImagePrev = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _inputWidth, _inputHeight);

	{
		cl_uint4 clear = { 0, 0, 0, 0 };

		cl::size_t<3> origin;
		origin[0] = 0;
		origin[1] = 0;
		origin[2] = 0;

		cl::size_t<3> region;
		region[0] = _inputWidth;
		region[1] = _inputHeight;
		region[2] = 1;

		cs.getQueue().enqueueFillImage(_inputImage, clear, origin, region);
		cs.getQueue().enqueueFillImage(_inputImagePrev, clear, origin, region);
	}

	int prevWidth = _inputWidth;
	int prevHeight = _inputHeight;

	for (int l = 0; l < _layers.size(); l++) {
		int numSpatialWeights = std::pow(_layerDescs[l]._receptiveFieldRadius * 2 + 1, 2) + 1; // + 1 for bias
		int numReconstructionWeights = std::pow(_layerDescs[l]._reconstructionRadius * 2 + 1, 2) + 1; // + 1 for bias
		int numPredictiveWeights = std::pow(_layerDescs[l]._predictiveRadius * 2 + 1, 2) + 1; // + 1 for bias
		int numLateralWeights = std::pow(_layerDescs[l]._lateralConnectionRadius * 2 + 1, 2);
		int numFeedBackWeights = std::pow(_layerDescs[l]._feedBackConnectionRadius * 2 + 1, 2);

		_layers[l]._hiddenActivationsSpatial = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _layerDescs[l]._spatialWidth, _layerDescs[l]._spatialHeight);
		_layers[l]._hiddenStatesSpatial = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), _layerDescs[l]._spatialWidth, _layerDescs[l]._spatialHeight);
		_layers[l]._hiddenStatesSpatialPrev = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), _layerDescs[l]._spatialWidth, _layerDescs[l]._spatialHeight);

		_layers[l]._hiddenActivationsTemporal = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _layerDescs[l]._temporalWidth, _layerDescs[l]._temporalHeight);
		_layers[l]._hiddenStatesTemporal = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), _layerDescs[l]._temporalWidth, _layerDescs[l]._temporalHeight);
		_layers[l]._hiddenStatesTemporalPrev = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), _layerDescs[l]._temporalWidth, _layerDescs[l]._temporalHeight);
		_layers[l]._hiddenStatesTemporalPrevPrev = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), _layerDescs[l]._temporalWidth, _layerDescs[l]._temporalHeight);

		_layers[l]._spatialWeights = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), _layerDescs[l]._spatialWidth, _layerDescs[l]._spatialHeight, numSpatialWeights);
		_layers[l]._spatialWeightsPrev = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), _layerDescs[l]._spatialWidth, _layerDescs[l]._spatialHeight, numSpatialWeights);

		_layers[l]._spatialPredictiveReconstructionWeights = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), _layerDescs[l]._spatialWidth, _layerDescs[l]._spatialHeight, numReconstructionWeights);
		_layers[l]._spatialPredictiveReconstructionWeightsPrev = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), _layerDescs[l]._spatialWidth, _layerDescs[l]._spatialHeight, numReconstructionWeights);

		_layers[l]._predictiveWeights = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), _layerDescs[l]._temporalWidth, _layerDescs[l]._temporalHeight, numPredictiveWeights);
		_layers[l]._predictiveWeightsPrev = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), _layerDescs[l]._temporalWidth, _layerDescs[l]._temporalHeight, numPredictiveWeights);

		_layers[l]._lateralWeights = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), _layerDescs[l]._temporalWidth, _layerDescs[l]._temporalHeight, numLateralWeights);
		_layers[l]._lateralWeightsPrev = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), _layerDescs[l]._temporalWidth, _layerDescs[l]._temporalHeight, numLateralWeights);

		_layers[l]._feedBackWeights = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), _layerDescs[l]._temporalWidth, _layerDescs[l]._temporalHeight, numFeedBackWeights);
		_layers[l]._feedBackWeightsPrev = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), _layerDescs[l]._temporalWidth, _layerDescs[l]._temporalHeight, numFeedBackWeights);

		_layers[l]._spatialReconstruction = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _layerDescs[l]._spatialWidth, _layerDescs[l]._spatialHeight);
		_layers[l]._spatialReconstructionPrev = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _layerDescs[l]._spatialWidth, _layerDescs[l]._spatialHeight);

		_layers[l]._temporalReconstruction = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _layerDescs[l]._temporalWidth, _layerDescs[l]._temporalHeight);
		_layers[l]._temporalReconstructionPrev = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _layerDescs[l]._temporalWidth, _layerDescs[l]._temporalHeight);

		if (l != _layers.size() - 1) {
			_layers[l]._nextTemporalReconstruction = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _layerDescs[l + 1]._temporalWidth, _layerDescs[l + 1]._temporalHeight);
			_layers[l]._nextTemporalReconstructionPrev = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _layerDescs[l + 1]._temporalWidth, _layerDescs[l + 1]._temporalHeight);
		}

		_layers[l]._predictedSpatial = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _layerDescs[l]._spatialWidth, _layerDescs[l]._spatialHeight);
		_layers[l]._predictedSpatialPrev = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _layerDescs[l]._spatialWidth, _layerDescs[l]._spatialHeight);

		_layers[l]._inputReconstruction = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), prevWidth, prevHeight);

		_layers[l]._predictedInputReconstruction = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), prevWidth, prevHeight);

		// Initialize
		Uint2 initSeedHiddenSpatial;
		initSeedHiddenSpatial._x = seedDist(generator);
		initSeedHiddenSpatial._y = seedDist(generator);

		int index = 0;

		initializeLayerHiddenSpatialKernel.setArg(index++, _layers[l]._spatialWeights);
		initializeLayerHiddenSpatialKernel.setArg(index++, _layers[l]._spatialPredictiveReconstructionWeights);
		initializeLayerHiddenSpatialKernel.setArg(index++, _layers[l]._hiddenStatesSpatial);
		initializeLayerHiddenSpatialKernel.setArg(index++, numSpatialWeights);
		initializeLayerHiddenSpatialKernel.setArg(index++, numReconstructionWeights);
		initializeLayerHiddenSpatialKernel.setArg(index++, initSeedHiddenSpatial);
		initializeLayerHiddenSpatialKernel.setArg(index++, _layerDescs[l]._spatialSparsity);
		initializeLayerHiddenSpatialKernel.setArg(index++, minInitWeight);
		initializeLayerHiddenSpatialKernel.setArg(index++, maxInitWeight);

		cs.getQueue().enqueueNDRangeKernel(initializeLayerHiddenSpatialKernel, cl::NullRange, cl::NDRange(_layerDescs[l]._spatialWidth, _layerDescs[l]._spatialHeight));

		Uint2 initSeedHiddenTemporal;
		initSeedHiddenTemporal._x = seedDist(generator);
		initSeedHiddenTemporal._y = seedDist(generator);

		index = 0;

		initializeLayerHiddenTemporalKernel.setArg(index++, _layers[l]._predictiveWeights);
		initializeLayerHiddenTemporalKernel.setArg(index++, _layers[l]._lateralWeights);
		initializeLayerHiddenTemporalKernel.setArg(index++, _layers[l]._feedBackWeights);
		initializeLayerHiddenTemporalKernel.setArg(index++, _layers[l]._hiddenStatesTemporal);
		initializeLayerHiddenTemporalKernel.setArg(index++, numPredictiveWeights);
		initializeLayerHiddenTemporalKernel.setArg(index++, numLateralWeights);
		initializeLayerHiddenTemporalKernel.setArg(index++, numFeedBackWeights);
		initializeLayerHiddenTemporalKernel.setArg(index++, initSeedHiddenTemporal);
		initializeLayerHiddenTemporalKernel.setArg(index++, _layerDescs[l]._temporalSparsity);
		initializeLayerHiddenTemporalKernel.setArg(index++, _layerDescs[l]._lateralScalar);
		initializeLayerHiddenTemporalKernel.setArg(index++, _layerDescs[l]._feedBackScalar);
		initializeLayerHiddenTemporalKernel.setArg(index++, minInitWeight);
		initializeLayerHiddenTemporalKernel.setArg(index++, maxInitWeight);

		cs.getQueue().enqueueNDRangeKernel(initializeLayerHiddenTemporalKernel, cl::NullRange, cl::NDRange(_layerDescs[l]._temporalWidth, _layerDescs[l]._temporalHeight));


		{
			cl::size_t<3> origin;
			origin[0] = 0;
			origin[1] = 0;
			origin[2] = 0;

			cl::size_t<3> region;
			region[0] = _layerDescs[l]._spatialWidth;
			region[1] = _layerDescs[l]._spatialHeight;
			region[2] = numSpatialWeights;

			cs.getQueue().enqueueCopyImage(_layers[l]._spatialWeights, _layers[l]._spatialWeightsPrev, origin, origin, region);
		}

		{
			cl::size_t<3> origin;
			origin[0] = 0;
			origin[1] = 0;
			origin[2] = 0;

			cl::size_t<3> region;
			region[0] = _layerDescs[l]._spatialWidth;
			region[1] = _layerDescs[l]._spatialHeight;
			region[2] = 1;

			cs.getQueue().enqueueCopyImage(_layers[l]._hiddenStatesSpatial, _layers[l]._hiddenStatesSpatialPrev, origin, origin, region);
		}

		{
			cl::size_t<3> origin;
			origin[0] = 0;
			origin[1] = 0;
			origin[2] = 0;

			cl::size_t<3> region;
			region[0] = _layerDescs[l]._temporalWidth;
			region[1] = _layerDescs[l]._temporalHeight;
			region[2] = numPredictiveWeights;

			cs.getQueue().enqueueCopyImage(_layers[l]._predictiveWeights, _layers[l]._predictiveWeightsPrev, origin, origin, region);
		}

		{
			cl::size_t<3> origin;
			origin[0] = 0;
			origin[1] = 0;
			origin[2] = 0;

			cl::size_t<3> region;
			region[0] = _layerDescs[l]._temporalWidth;
			region[1] = _layerDescs[l]._temporalHeight;
			region[2] = numLateralWeights;

			cs.getQueue().enqueueCopyImage(_layers[l]._lateralWeights, _layers[l]._lateralWeightsPrev, origin, origin, region);
		}

		{
			cl::size_t<3> origin;
			origin[0] = 0;
			origin[1] = 0;
			origin[2] = 0;

			cl::size_t<3> region;
			region[0] = _layerDescs[l]._temporalWidth;
			region[1] = _layerDescs[l]._temporalHeight;
			region[2] = numFeedBackWeights;

			cs.getQueue().enqueueCopyImage(_layers[l]._feedBackWeights, _layers[l]._feedBackWeightsPrev, origin, origin, region);
		}

		{
			cl::size_t<3> origin;
			origin[0] = 0;
			origin[1] = 0;
			origin[2] = 0;

			cl::size_t<3> region;
			region[0] = _layerDescs[l]._temporalWidth;
			region[1] = _layerDescs[l]._temporalHeight;
			region[2] = 1;

			cs.getQueue().enqueueCopyImage(_layers[l]._hiddenStatesTemporal, _layers[l]._hiddenStatesTemporalPrev, origin, origin, region);
			cs.getQueue().enqueueCopyImage(_layers[l]._hiddenStatesTemporal, _layers[l]._hiddenStatesTemporalPrevPrev, origin, origin, region);
		}

		
		prevWidth = _layerDescs[l]._spatialWidth;
		prevHeight = _layerDescs[l]._spatialHeight;
	}

	_layerInhibitKernel = cl::Kernel(program.getProgram(), "layerInhibit");
	_layerHiddenStatesSpatialActivateKernel = cl::Kernel(program.getProgram(), "layerHiddenStatesSpatialActivate");
	_layerHiddenStatesTemporalActivateKernel = cl::Kernel(program.getProgram(), "layerHiddenStatesTemporalActivate");
	_layerHiddenStatesTemporalActivateLastKernel = cl::Kernel(program.getProgram(), "layerHiddenStatesTemporalActivateLast");
	_layerInputReconstructKernel = cl::Kernel(program.getProgram(), "layerInputReconstruct");
	_layerInputReconstructLinearKernel = cl::Kernel(program.getProgram(), "layerInputReconstructLinear");
	_layerSpatialReconstructKernel = cl::Kernel(program.getProgram(), "layerSpatialReconstruct");
	_layerTemporalReconstructKernel = cl::Kernel(program.getProgram(), "layerTemporalReconstruct");
	_layerNextTemporalReconstructKernel = cl::Kernel(program.getProgram(), "layerNextTemporalReconstruct");
	_layerUpdateSpatialWeightsKernel = cl::Kernel(program.getProgram(), "layerUpdateSpatialWeights");
	_layerSpatialPredictiveReconstructKernel = cl::Kernel(program.getProgram(), "layerSpatialPredictiveReconstruct");
	_layerUpdateTemporalWeightsKernel = cl::Kernel(program.getProgram(), "layerUpdateTemporalWeights");
	_layerUpdateTemporalWeightsLastKernel = cl::Kernel(program.getProgram(), "layerUpdateTemporalWeightsLast");
	_layerSpatialPredictiveReconstructionWeightUpdateKernel = cl::Kernel(program.getProgram(), "layerSpatialPredictiveReconstructionWeightUpdate");
	_gaussianBlurXKernel = cl::Kernel(program.getProgram(), "gaussianBlurX");
	_gaussianBlurYKernel = cl::Kernel(program.getProgram(), "gaussianBlurY");
}

void HTFE::activate(sys::ComputeSystem &cs, std::mt19937 &generator) {
	{
		cl::size_t<3> origin;
		origin[0] = 0;
		origin[1] = 0;
		origin[2] = 0;

		cl::size_t<3> region;
		region[0] = _inputWidth;
		region[1] = _inputHeight;
		region[2] = 1;

		cs.getQueue().enqueueWriteImage(_inputImage, CL_TRUE, origin, region, 0, 0, _input.data());
	}

	std::uniform_int_distribution<int> seedDist(0, 99999);

	// ------------------------------------------------------------------------------
	// ------------------------------------ Go up -----------------------------------
	// ------------------------------------------------------------------------------

	cl::Image2D* pPrevLayer = &_inputImage;
	int prevWidth = _inputWidth;
	int prevHeight = _inputHeight;

	for (int l = 0; l < _layers.size(); l++) {
		float localActivity = std::ceil(_layerDescs[l]._spatialSparsity * std::pow(_layerDescs[l]._spatialInhibitionRadius * 2 + 1, 2));

		Int2 layerSizeSpatial;
		layerSizeSpatial._x = _layerDescs[l]._spatialWidth;
		layerSizeSpatial._y = _layerDescs[l]._spatialHeight;

		Int2 layerSizeSpatialMinusOne;
		layerSizeSpatialMinusOne._x = _layerDescs[l]._spatialWidth - 1;
		layerSizeSpatialMinusOne._y = _layerDescs[l]._spatialHeight - 1;

		Float2 layerSizeSpatialMinusOneInv;
		layerSizeSpatialMinusOneInv._x = 1.0f / (_layerDescs[l]._spatialWidth - 1);
		layerSizeSpatialMinusOneInv._y = 1.0f / (_layerDescs[l]._spatialHeight - 1);

		Int2 layerSizeTemporal;
		layerSizeTemporal._x = _layerDescs[l]._temporalWidth;
		layerSizeTemporal._y = _layerDescs[l]._temporalHeight;

		Int2 layerSizeTemporalMinusOne;
		layerSizeTemporalMinusOne._x = _layerDescs[l]._temporalWidth - 1;
		layerSizeTemporalMinusOne._y = _layerDescs[l]._temporalHeight - 1;

		Float2 layerSizeTemporalMinusOneInv;
		layerSizeTemporalMinusOneInv._x = 1.0f / (_layerDescs[l]._temporalWidth - 1);
		layerSizeTemporalMinusOneInv._y = 1.0f / (_layerDescs[l]._temporalHeight - 1);

		Int2 inputSize;
		inputSize._x = prevWidth;
		inputSize._y = prevHeight;

		Int2 inputSizeMinusOne;
		inputSizeMinusOne._x = prevWidth - 1;
		inputSizeMinusOne._y = prevHeight - 1;

		Float2 inputSizeMinusOneInv;
		inputSizeMinusOneInv._x = 1.0f / (prevWidth - 1);
		inputSizeMinusOneInv._y = 1.0f / (prevHeight - 1);

		Int2 reverseReceptiveRadius;
		reverseReceptiveRadius._x = std::ceil(static_cast<float>(_layerDescs[l]._spatialWidth) / static_cast<float>(prevWidth)* static_cast<float>(_layerDescs.front()._receptiveFieldRadius));
		reverseReceptiveRadius._y = std::ceil(static_cast<float>(_layerDescs[l]._spatialHeight) / static_cast<float>(prevHeight)* static_cast<float>(_layerDescs.front()._receptiveFieldRadius));

		// -------------------------------- Activate --------------------------------

		Uint2 activateFeedForwardSeed;
		activateFeedForwardSeed._x = seedDist(generator);
		activateFeedForwardSeed._y = seedDist(generator);

		int index = 0;

		_layerHiddenStatesSpatialActivateKernel.setArg(index++, *pPrevLayer);
		_layerHiddenStatesSpatialActivateKernel.setArg(index++, _layers[l]._spatialWeightsPrev);
		_layerHiddenStatesSpatialActivateKernel.setArg(index++, _layers[l]._hiddenActivationsSpatial);
		_layerHiddenStatesSpatialActivateKernel.setArg(index++, layerSizeSpatial);
		_layerHiddenStatesSpatialActivateKernel.setArg(index++, layerSizeSpatialMinusOneInv);
		_layerHiddenStatesSpatialActivateKernel.setArg(index++, inputSize);
		_layerHiddenStatesSpatialActivateKernel.setArg(index++, inputSizeMinusOne);
		_layerHiddenStatesSpatialActivateKernel.setArg(index++, _layerDescs[l]._receptiveFieldRadius);
		_layerHiddenStatesSpatialActivateKernel.setArg(index++, _layerDescs[l]._dutyCycleDecay);
		_layerHiddenStatesSpatialActivateKernel.setArg(index++, _layerDescs[l]._minDerivative);
		_layerHiddenStatesSpatialActivateKernel.setArg(index++, activateFeedForwardSeed);

		cs.getQueue().enqueueNDRangeKernel(_layerHiddenStatesSpatialActivateKernel, cl::NullRange, cl::NDRange(_layerDescs[l]._spatialWidth, _layerDescs[l]._spatialHeight));

		index = 0;

		_layerInhibitKernel.setArg(index++, _layers[l]._hiddenActivationsSpatial);
		_layerInhibitKernel.setArg(index++, _layers[l]._hiddenStatesSpatial);
		_layerInhibitKernel.setArg(index++, layerSizeSpatial);
		_layerInhibitKernel.setArg(index++, _layerDescs[l]._spatialInhibitionRadius);
		_layerInhibitKernel.setArg(index++, localActivity);
		_layerInhibitKernel.setArg(index++, _layerDescs[l]._minDerivative);

		cs.getQueue().enqueueNDRangeKernel(_layerInhibitKernel, cl::NullRange, cl::NDRange(_layerDescs[l]._spatialWidth, _layerDescs[l]._spatialHeight));

		// Reconstruction
		Uint2 seed;
		seed._x = seedDist(generator);
		seed._y = seedDist(generator);

		index = 0;

		if (l == 0) {
			_layerInputReconstructLinearKernel.setArg(index++, _layers[l]._hiddenStatesSpatial);
			_layerInputReconstructLinearKernel.setArg(index++, _layers[l]._spatialWeightsPrev);
			_layerInputReconstructLinearKernel.setArg(index++, _layers[l]._inputReconstruction);
			_layerInputReconstructLinearKernel.setArg(index++, _layerDescs[l]._receptiveFieldRadius);
			_layerInputReconstructLinearKernel.setArg(index++, reverseReceptiveRadius);
			_layerInputReconstructLinearKernel.setArg(index++, inputSizeMinusOne);
			_layerInputReconstructLinearKernel.setArg(index++, inputSizeMinusOneInv);
			_layerInputReconstructLinearKernel.setArg(index++, layerSizeSpatial);
			_layerInputReconstructLinearKernel.setArg(index++, layerSizeSpatialMinusOne);
			_layerInputReconstructLinearKernel.setArg(index++, layerSizeSpatialMinusOneInv);
			_layerInputReconstructLinearKernel.setArg(index++, seed);
			_layerInputReconstructLinearKernel.setArg(index++, _layerDescs[l]._gaussianNoise);

			cs.getQueue().enqueueNDRangeKernel(_layerInputReconstructLinearKernel, cl::NullRange, cl::NDRange(prevWidth, prevHeight));
		}
		else {
			_layerInputReconstructKernel.setArg(index++, _layers[l]._hiddenStatesSpatial);
			_layerInputReconstructKernel.setArg(index++, _layers[l]._spatialWeightsPrev);
			_layerInputReconstructKernel.setArg(index++, _layers[l]._inputReconstruction);
			_layerInputReconstructKernel.setArg(index++, _layerDescs[l]._receptiveFieldRadius);
			_layerInputReconstructKernel.setArg(index++, reverseReceptiveRadius);
			_layerInputReconstructKernel.setArg(index++, inputSizeMinusOne);
			_layerInputReconstructKernel.setArg(index++, inputSizeMinusOneInv);
			_layerInputReconstructKernel.setArg(index++, layerSizeSpatial);
			_layerInputReconstructKernel.setArg(index++, layerSizeSpatialMinusOne);
			_layerInputReconstructKernel.setArg(index++, layerSizeSpatialMinusOneInv);
			_layerInputReconstructKernel.setArg(index++, seed);

			cs.getQueue().enqueueNDRangeKernel(_layerInputReconstructKernel, cl::NullRange, cl::NDRange(prevWidth, prevHeight));
		}

		pPrevLayer = &_layers[l]._hiddenStatesSpatial;
		prevWidth = _layerDescs[l]._spatialWidth;
		prevHeight = _layerDescs[l]._spatialHeight;
	}

	// ------------------------------------------------------------------------------
	// -------------------------------- Go back down --------------------------------
	// ------------------------------------------------------------------------------

	for (int l = _layers.size() - 1; l >= 0; l--) {
		float localActivity = std::ceil(_layerDescs[l]._temporalSparsity * std::pow(_layerDescs[l]._temporalInhibitionRadius * 2 + 1, 2));

		if (l > 0) {
			pPrevLayer = &_layers[l - 1]._hiddenStatesSpatial;
			prevWidth = _layerDescs[l - 1]._spatialWidth;
			prevHeight = _layerDescs[l - 1]._spatialHeight;
		}
		else {
			pPrevLayer = &_inputImage;
			prevWidth = _inputWidth;
			prevHeight = _inputHeight;
		}

		Int2 layerSizeSpatial;
		layerSizeSpatial._x = _layerDescs[l]._spatialWidth;
		layerSizeSpatial._y = _layerDescs[l]._spatialHeight;

		Int2 layerSizeSpatialMinusOne;
		layerSizeSpatialMinusOne._x = _layerDescs[l]._spatialWidth - 1;
		layerSizeSpatialMinusOne._y = _layerDescs[l]._spatialHeight - 1;

		Float2 layerSizeSpatialMinusOneInv;
		layerSizeSpatialMinusOneInv._x = 1.0f / (_layerDescs[l]._spatialWidth - 1);
		layerSizeSpatialMinusOneInv._y = 1.0f / (_layerDescs[l]._spatialHeight - 1);

		Int2 layerSizeTemporal;
		layerSizeTemporal._x = _layerDescs[l]._temporalWidth;
		layerSizeTemporal._y = _layerDescs[l]._temporalHeight;

		Int2 layerSizeTemporalMinusOne;
		layerSizeTemporalMinusOne._x = _layerDescs[l]._temporalWidth - 1;
		layerSizeTemporalMinusOne._y = _layerDescs[l]._temporalHeight - 1;

		Float2 layerSizeTemporalMinusOneInv;
		layerSizeTemporalMinusOneInv._x = 1.0f / (_layerDescs[l]._temporalWidth - 1);
		layerSizeTemporalMinusOneInv._y = 1.0f / (_layerDescs[l]._temporalHeight - 1);

		Int2 inputSize;
		inputSize._x = prevWidth;
		inputSize._y = prevHeight;

		Int2 inputSizeMinusOne;
		inputSizeMinusOne._x = prevWidth - 1;
		inputSizeMinusOne._y = prevHeight - 1;

		Float2 inputSizeMinusOneInv;
		inputSizeMinusOneInv._x = 1.0f / (prevWidth - 1);
		inputSizeMinusOneInv._y = 1.0f / (prevHeight - 1);

		Int2 reversePredictiveRadius;
		reversePredictiveRadius._x = std::ceil(static_cast<float>(_layerDescs[l]._temporalWidth) / static_cast<float>(_layerDescs[l]._spatialWidth) * static_cast<float>(_layerDescs.front()._predictiveRadius));
		reversePredictiveRadius._y = std::ceil(static_cast<float>(_layerDescs[l]._temporalHeight) / static_cast<float>(_layerDescs[l]._spatialHeight) * static_cast<float>(_layerDescs.front()._predictiveRadius));

		Int2 reverseReceptiveRadius;
		reverseReceptiveRadius._x = std::ceil(static_cast<float>(_layerDescs[l]._spatialWidth) / static_cast<float>(prevWidth) * static_cast<float>(_layerDescs.front()._receptiveFieldRadius));
		reverseReceptiveRadius._y = std::ceil(static_cast<float>(_layerDescs[l]._spatialHeight) / static_cast<float>(prevHeight) * static_cast<float>(_layerDescs.front()._receptiveFieldRadius));

		Int2 nextTemporalSize;
		Int2 nextTemporalSizeMinusOne;
		Float2 nextTemporalSizeMinusOneInv;

		if (l == _layers.size() - 1) {
			nextTemporalSize._x = nextTemporalSize._y = 1;
			nextTemporalSizeMinusOne._x = nextTemporalSizeMinusOne._y = 0;
			nextTemporalSizeMinusOneInv._x = nextTemporalSizeMinusOneInv._y = 1.0f;
		}
		else {
			nextTemporalSize._x = _layerDescs[l + 1]._temporalWidth;
			nextTemporalSize._y = _layerDescs[l + 1]._temporalHeight;
			nextTemporalSizeMinusOne._x = _layerDescs[l + 1]._temporalWidth - 1;
			nextTemporalSizeMinusOne._y = _layerDescs[l + 1]._temporalHeight - 1;
			nextTemporalSizeMinusOneInv._x = 1.0f / nextTemporalSizeMinusOne._x;
			nextTemporalSizeMinusOneInv._y = 1.0f / nextTemporalSizeMinusOne._y;
		}

		Int2 reverseFeedBackRadius;
		reverseFeedBackRadius._x = std::ceil(static_cast<float>(_layerDescs[l]._temporalWidth) / static_cast<float>(nextTemporalSize._x)* static_cast<float>(_layerDescs.front()._feedBackConnectionRadius));
		reverseFeedBackRadius._y = std::ceil(static_cast<float>(_layerDescs[l]._temporalHeight) / static_cast<float>(nextTemporalSize._y)* static_cast<float>(_layerDescs.front()._feedBackConnectionRadius));

		// -------------------------------- Activate --------------------------------

		Uint2 activateFeedBackSeed;
		activateFeedBackSeed._x = seedDist(generator);
		activateFeedBackSeed._y = seedDist(generator);

		int index = 0;

		if (l == _layers.size() - 1) {
			_layerHiddenStatesTemporalActivateLastKernel.setArg(index++, _layers[l]._hiddenStatesSpatial);
			_layerHiddenStatesTemporalActivateLastKernel.setArg(index++, _layers[l]._hiddenStatesTemporalPrev);
			_layerHiddenStatesTemporalActivateLastKernel.setArg(index++, _layers[l]._predictiveWeightsPrev);
			_layerHiddenStatesTemporalActivateLastKernel.setArg(index++, _layers[l]._lateralWeightsPrev);
			_layerHiddenStatesTemporalActivateLastKernel.setArg(index++, _layers[l]._hiddenActivationsTemporal);
			_layerHiddenStatesTemporalActivateLastKernel.setArg(index++, layerSizeTemporal);
			_layerHiddenStatesTemporalActivateLastKernel.setArg(index++, layerSizeTemporalMinusOneInv);
			_layerHiddenStatesTemporalActivateLastKernel.setArg(index++, layerSizeSpatial);
			_layerHiddenStatesTemporalActivateLastKernel.setArg(index++, layerSizeSpatialMinusOne);
			_layerHiddenStatesTemporalActivateLastKernel.setArg(index++, _layerDescs[l]._predictiveRadius);
			_layerHiddenStatesTemporalActivateLastKernel.setArg(index++, _layerDescs[l]._lateralConnectionRadius);
			_layerHiddenStatesTemporalActivateLastKernel.setArg(index++, _layerDescs[l]._dutyCycleDecay);
			_layerHiddenStatesTemporalActivateLastKernel.setArg(index++, _layerDescs[l]._minDerivative);
			_layerHiddenStatesTemporalActivateLastKernel.setArg(index++, activateFeedBackSeed);

			cs.getQueue().enqueueNDRangeKernel(_layerHiddenStatesTemporalActivateLastKernel, cl::NullRange, cl::NDRange(_layerDescs[l]._temporalWidth, _layerDescs[l]._temporalHeight));
		}
		else {
			_layerHiddenStatesTemporalActivateKernel.setArg(index++, _layers[l]._hiddenStatesSpatial);
			_layerHiddenStatesTemporalActivateKernel.setArg(index++, _layers[l]._hiddenStatesTemporalPrev);
			_layerHiddenStatesTemporalActivateKernel.setArg(index++, _layers[l + 1]._hiddenStatesTemporal);
			_layerHiddenStatesTemporalActivateKernel.setArg(index++, _layers[l]._predictiveWeightsPrev);
			_layerHiddenStatesTemporalActivateKernel.setArg(index++, _layers[l]._feedBackWeightsPrev);
			_layerHiddenStatesTemporalActivateKernel.setArg(index++, _layers[l]._lateralWeightsPrev);
			_layerHiddenStatesTemporalActivateKernel.setArg(index++, _layers[l]._hiddenActivationsTemporal);
			_layerHiddenStatesTemporalActivateKernel.setArg(index++, layerSizeTemporal);
			_layerHiddenStatesTemporalActivateKernel.setArg(index++, layerSizeTemporalMinusOneInv);
			_layerHiddenStatesTemporalActivateKernel.setArg(index++, layerSizeSpatial);
			_layerHiddenStatesTemporalActivateKernel.setArg(index++, layerSizeSpatialMinusOne);
			_layerHiddenStatesTemporalActivateKernel.setArg(index++, nextTemporalSize);
			_layerHiddenStatesTemporalActivateKernel.setArg(index++, nextTemporalSizeMinusOne);
			_layerHiddenStatesTemporalActivateKernel.setArg(index++, _layerDescs[l]._predictiveRadius);
			_layerHiddenStatesTemporalActivateKernel.setArg(index++, _layerDescs[l]._feedBackConnectionRadius);
			_layerHiddenStatesTemporalActivateKernel.setArg(index++, _layerDescs[l]._lateralConnectionRadius);
			_layerHiddenStatesTemporalActivateKernel.setArg(index++, _layerDescs[l]._dutyCycleDecay);
			_layerHiddenStatesTemporalActivateKernel.setArg(index++, _layerDescs[l]._minDerivative);
			_layerHiddenStatesTemporalActivateKernel.setArg(index++, activateFeedBackSeed);

			cs.getQueue().enqueueNDRangeKernel(_layerHiddenStatesTemporalActivateKernel, cl::NullRange, cl::NDRange(_layerDescs[l]._temporalWidth, _layerDescs[l]._temporalHeight));
		}

		index = 0;

		_layerInhibitKernel.setArg(index++, _layers[l]._hiddenActivationsTemporal);
		_layerInhibitKernel.setArg(index++, _layers[l]._hiddenStatesTemporal);
		_layerInhibitKernel.setArg(index++, layerSizeTemporal);
		_layerInhibitKernel.setArg(index++, _layerDescs[l]._temporalInhibitionRadius);
		_layerInhibitKernel.setArg(index++, localActivity);
		_layerInhibitKernel.setArg(index++, _layerDescs[l]._minDerivative);

		cs.getQueue().enqueueNDRangeKernel(_layerInhibitKernel, cl::NullRange, cl::NDRange(_layerDescs[l]._temporalWidth, _layerDescs[l]._temporalHeight));

		// --------------------- Reconstructions ---------------------

		Uint2 seed;
		seed._x = seedDist(generator);
		seed._y = seedDist(generator);

		index = 0;

		_layerSpatialReconstructKernel.setArg(index++, _layers[l]._hiddenStatesTemporal);
		_layerSpatialReconstructKernel.setArg(index++, _layers[l]._predictiveWeightsPrev);
		_layerSpatialReconstructKernel.setArg(index++, _layers[l]._spatialReconstruction);
		_layerSpatialReconstructKernel.setArg(index++, _layerDescs[l]._predictiveRadius);
		_layerSpatialReconstructKernel.setArg(index++, reversePredictiveRadius);
		_layerSpatialReconstructKernel.setArg(index++, layerSizeSpatial);
		_layerSpatialReconstructKernel.setArg(index++, layerSizeSpatialMinusOne);
		_layerSpatialReconstructKernel.setArg(index++, layerSizeSpatialMinusOneInv);
		_layerSpatialReconstructKernel.setArg(index++, layerSizeTemporal);
		_layerSpatialReconstructKernel.setArg(index++, layerSizeTemporalMinusOne);
		_layerSpatialReconstructKernel.setArg(index++, layerSizeTemporalMinusOneInv);
		_layerSpatialReconstructKernel.setArg(index++, seed);

		cs.getQueue().enqueueNDRangeKernel(_layerSpatialReconstructKernel, cl::NullRange, cl::NDRange(_layerDescs[l]._spatialWidth, _layerDescs[l]._spatialHeight));

		seed._x = seedDist(generator);
		seed._y = seedDist(generator);

		index = 0;

		_layerTemporalReconstructKernel.setArg(index++, _layers[l]._hiddenStatesTemporal);
		_layerTemporalReconstructKernel.setArg(index++, _layers[l]._lateralWeightsPrev);
		_layerTemporalReconstructKernel.setArg(index++, _layers[l]._temporalReconstruction);
		_layerTemporalReconstructKernel.setArg(index++, _layerDescs[l]._lateralConnectionRadius);
		_layerTemporalReconstructKernel.setArg(index++, layerSizeTemporal);
		_layerTemporalReconstructKernel.setArg(index++, seed);

		cs.getQueue().enqueueNDRangeKernel(_layerTemporalReconstructKernel, cl::NullRange, cl::NDRange(_layerDescs[l]._temporalWidth, _layerDescs[l]._temporalHeight));
	
		index = 0;

		if (l != _layers.size() - 1) {
			seed._x = seedDist(generator);
			seed._y = seedDist(generator);

			_layerNextTemporalReconstructKernel.setArg(index++, _layers[l]._hiddenStatesTemporal);
			_layerNextTemporalReconstructKernel.setArg(index++, _layers[l]._feedBackWeightsPrev);
			_layerNextTemporalReconstructKernel.setArg(index++, _layers[l]._nextTemporalReconstruction);
			_layerNextTemporalReconstructKernel.setArg(index++, _layerDescs[l]._feedBackConnectionRadius);
			_layerNextTemporalReconstructKernel.setArg(index++, reverseFeedBackRadius);
			_layerNextTemporalReconstructKernel.setArg(index++, nextTemporalSizeMinusOne);
			_layerNextTemporalReconstructKernel.setArg(index++, nextTemporalSizeMinusOneInv);
			_layerNextTemporalReconstructKernel.setArg(index++, layerSizeTemporal);
			_layerNextTemporalReconstructKernel.setArg(index++, layerSizeTemporalMinusOne);
			_layerNextTemporalReconstructKernel.setArg(index++, layerSizeTemporalMinusOneInv);
			_layerNextTemporalReconstructKernel.setArg(index++, seed);

			cs.getQueue().enqueueNDRangeKernel(_layerNextTemporalReconstructKernel, cl::NullRange, cl::NDRange(nextTemporalSize._x, nextTemporalSize._y));
		}

		// --------------------- Predictive Spatial Reconstruction ---------------------

		seed._x = seedDist(generator);
		seed._y = seedDist(generator);

		index = 0;

		_layerSpatialPredictiveReconstructKernel.setArg(index++, _layers[l]._hiddenStatesTemporal);
		_layerSpatialPredictiveReconstructKernel.setArg(index++, _layers[l]._spatialPredictiveReconstructionWeightsPrev);
		_layerSpatialPredictiveReconstructKernel.setArg(index++, _layers[l]._predictedSpatial);
		_layerSpatialPredictiveReconstructKernel.setArg(index++, _layerDescs[l]._reconstructionRadius);
		_layerSpatialPredictiveReconstructKernel.setArg(index++, layerSizeSpatialMinusOne);
		_layerSpatialPredictiveReconstructKernel.setArg(index++, layerSizeSpatialMinusOneInv);
		_layerSpatialPredictiveReconstructKernel.setArg(index++, layerSizeTemporal);
		_layerSpatialPredictiveReconstructKernel.setArg(index++, layerSizeTemporalMinusOne);
		_layerSpatialPredictiveReconstructKernel.setArg(index++, layerSizeTemporalMinusOneInv);
	
		cs.getQueue().enqueueNDRangeKernel(_layerSpatialPredictiveReconstructKernel, cl::NullRange, cl::NDRange(_layerDescs[l]._spatialWidth, _layerDescs[l]._spatialHeight));

		// --------------------- Input Reconstruction ---------------------

		seed._x = seedDist(generator);
		seed._y = seedDist(generator);

		index = 0;

		if (l == 0) {	
			_layerInputReconstructLinearKernel.setArg(index++, _layers[l]._predictedSpatial);
			_layerInputReconstructLinearKernel.setArg(index++, _layers[l]._spatialWeightsPrev);
			_layerInputReconstructLinearKernel.setArg(index++, _layers[l]._predictedInputReconstruction);
			_layerInputReconstructLinearKernel.setArg(index++, _layerDescs[l]._receptiveFieldRadius);
			_layerInputReconstructLinearKernel.setArg(index++, reverseReceptiveRadius);
			_layerInputReconstructLinearKernel.setArg(index++, inputSizeMinusOne);
			_layerInputReconstructLinearKernel.setArg(index++, inputSizeMinusOneInv);
			_layerInputReconstructLinearKernel.setArg(index++, layerSizeSpatial);
			_layerInputReconstructLinearKernel.setArg(index++, layerSizeSpatialMinusOne);
			_layerInputReconstructLinearKernel.setArg(index++, layerSizeSpatialMinusOneInv);
			_layerInputReconstructLinearKernel.setArg(index++, seed);
			_layerInputReconstructLinearKernel.setArg(index++, _layerDescs[l]._gaussianNoise);

			cs.getQueue().enqueueNDRangeKernel(_layerInputReconstructLinearKernel, cl::NullRange, cl::NDRange(prevWidth, prevHeight));
		}
		else {
			_layerInputReconstructKernel.setArg(index++, _layers[l]._predictedSpatial);
			_layerInputReconstructKernel.setArg(index++, _layers[l]._spatialWeightsPrev);
			_layerInputReconstructKernel.setArg(index++, _layers[l]._predictedInputReconstruction);
			_layerInputReconstructKernel.setArg(index++, _layerDescs[l]._receptiveFieldRadius);
			_layerInputReconstructKernel.setArg(index++, reverseReceptiveRadius);
			_layerInputReconstructKernel.setArg(index++, inputSizeMinusOne);
			_layerInputReconstructKernel.setArg(index++, inputSizeMinusOneInv);
			_layerInputReconstructKernel.setArg(index++, layerSizeSpatial);
			_layerInputReconstructKernel.setArg(index++, layerSizeSpatialMinusOne);
			_layerInputReconstructKernel.setArg(index++, layerSizeSpatialMinusOneInv);
			_layerInputReconstructKernel.setArg(index++, seed);

			cs.getQueue().enqueueNDRangeKernel(_layerInputReconstructKernel, cl::NullRange, cl::NDRange(prevWidth, prevHeight));
		}
	}

	{
		cl::size_t<3> origin;
		origin[0] = 0;
		origin[1] = 0;
		origin[2] = 0;

		cl::size_t<3> region;
		region[0] = _inputWidth;
		region[1] = _inputHeight;
		region[2] = 1;

		cs.getQueue().enqueueReadImage(_layers.front()._predictedInputReconstruction, CL_TRUE, origin, region, 0, 0, _prediction.data());
	}
}

void HTFE::learn(sys::ComputeSystem &cs) {
	// ------------------------------------------------------------------------------
	// ---------------------- Weight Update and Predictions  ------------------------
	// ------------------------------------------------------------------------------

	cl::Image2D* pPrevLayer = &_inputImage;
	int prevWidth = _inputWidth;
	int prevHeight = _inputHeight;

	for (int l = 0; l < _layers.size(); l++) {
		Int2 layerSizeSpatial;
		layerSizeSpatial._x = _layerDescs[l]._spatialWidth;
		layerSizeSpatial._y = _layerDescs[l]._spatialHeight;

		Int2 layerSizeSpatialMinusOne;
		layerSizeSpatialMinusOne._x = _layerDescs[l]._spatialWidth - 1;
		layerSizeSpatialMinusOne._y = _layerDescs[l]._spatialHeight - 1;

		Float2 layerSizeSpatialMinusOneInv;
		layerSizeSpatialMinusOneInv._x = 1.0f / (_layerDescs[l]._spatialWidth - 1);
		layerSizeSpatialMinusOneInv._y = 1.0f / (_layerDescs[l]._spatialHeight - 1);

		Int2 layerSizeTemporal;
		layerSizeTemporal._x = _layerDescs[l]._temporalWidth;
		layerSizeTemporal._y = _layerDescs[l]._temporalHeight;

		Int2 layerSizeTemporalMinusOne;
		layerSizeTemporalMinusOne._x = _layerDescs[l]._temporalWidth - 1;
		layerSizeTemporalMinusOne._y = _layerDescs[l]._temporalHeight - 1;

		Float2 layerSizeTemporalMinusOneInv;
		layerSizeTemporalMinusOneInv._x = 1.0f / (_layerDescs[l]._temporalWidth - 1);
		layerSizeTemporalMinusOneInv._y = 1.0f / (_layerDescs[l]._temporalHeight - 1);

		Int2 inputSize;
		inputSize._x = prevWidth;
		inputSize._y = prevHeight;

		Int2 inputSizeMinusOne;
		inputSizeMinusOne._x = prevWidth - 1;
		inputSizeMinusOne._y = prevHeight - 1;

		Float2 inputSizeMinusOneInv;
		inputSizeMinusOneInv._x = 1.0f / (prevWidth - 1);
		inputSizeMinusOneInv._y = 1.0f / (prevHeight - 1);

		Int2 reversePredictiveRadius;
		reversePredictiveRadius._x = std::ceil(static_cast<float>(_layerDescs[l]._temporalWidth) / static_cast<float>(_layerDescs[l]._spatialWidth) * static_cast<float>(_layerDescs.front()._predictiveRadius));
		reversePredictiveRadius._y = std::ceil(static_cast<float>(_layerDescs[l]._temporalHeight) / static_cast<float>(_layerDescs[l]._spatialHeight) * static_cast<float>(_layerDescs.front()._predictiveRadius));

		Int2 reverseReceptiveRadius;
		reverseReceptiveRadius._x = std::ceil(static_cast<float>(_layerDescs[l]._spatialWidth) / static_cast<float>(prevWidth)* static_cast<float>(_layerDescs.front()._receptiveFieldRadius));
		reverseReceptiveRadius._y = std::ceil(static_cast<float>(_layerDescs[l]._spatialHeight) / static_cast<float>(prevHeight)* static_cast<float>(_layerDescs.front()._receptiveFieldRadius));

		Int2 nextTemporalSize;
		Int2 nextTemporalSizeMinusOne;
		Float2 nextTemporalSizeMinusOneInv;

		if (l == _layers.size() - 1) {
			nextTemporalSize._x = nextTemporalSize._y = 1;
			nextTemporalSizeMinusOne._x = nextTemporalSizeMinusOne._y = 0;
			nextTemporalSizeMinusOneInv._x = nextTemporalSizeMinusOneInv._y = 1.0f;
		}
		else {
			nextTemporalSize._x = _layerDescs[l + 1]._temporalWidth;
			nextTemporalSize._y = _layerDescs[l + 1]._temporalHeight;
			nextTemporalSizeMinusOne._x = _layerDescs[l + 1]._temporalWidth - 1;
			nextTemporalSizeMinusOne._y = _layerDescs[l + 1]._temporalHeight - 1;
			nextTemporalSizeMinusOneInv._x = 1.0f / nextTemporalSizeMinusOne._x;
			nextTemporalSizeMinusOneInv._y = 1.0f / nextTemporalSizeMinusOne._y;
		}

		Int2 reverseFeedBackRadius;
		reverseFeedBackRadius._x = std::ceil(static_cast<float>(_layerDescs[l]._temporalWidth) / static_cast<float>(nextTemporalSize._x)* static_cast<float>(_layerDescs.front()._feedBackConnectionRadius));
		reverseFeedBackRadius._y = std::ceil(static_cast<float>(_layerDescs[l]._temporalHeight) / static_cast<float>(nextTemporalSize._y)* static_cast<float>(_layerDescs.front()._feedBackConnectionRadius));

		// ------------------------------- Weight Updates -------------------------------

		int index = 0;

		// Spatial reconstruction and weight update

		_layerUpdateSpatialWeightsKernel.setArg(index++, *pPrevLayer);
		_layerUpdateSpatialWeightsKernel.setArg(index++, _layers[l]._inputReconstruction);
		_layerUpdateSpatialWeightsKernel.setArg(index++, _layers[l]._hiddenStatesSpatial);
		_layerUpdateSpatialWeightsKernel.setArg(index++, _layers[l]._spatialWeightsPrev);
		_layerUpdateSpatialWeightsKernel.setArg(index++, _layers[l]._spatialWeights);
		_layerUpdateSpatialWeightsKernel.setArg(index++, layerSizeSpatial);
		_layerUpdateSpatialWeightsKernel.setArg(index++, layerSizeSpatialMinusOneInv);
		_layerUpdateSpatialWeightsKernel.setArg(index++, inputSize);
		_layerUpdateSpatialWeightsKernel.setArg(index++, inputSizeMinusOne);
		_layerUpdateSpatialWeightsKernel.setArg(index++, _layerDescs[l]._receptiveFieldRadius);
		_layerUpdateSpatialWeightsKernel.setArg(index++, _layerDescs[l]._spatialInhibitionRadius);
		_layerUpdateSpatialWeightsKernel.setArg(index++, _layerDescs[l]._spatialSparsity);
		_layerUpdateSpatialWeightsKernel.setArg(index++, _layerDescs[l]._spatialAlpha);
		_layerUpdateSpatialWeightsKernel.setArg(index++, _layerDescs[l]._spatialMomentum);
		_layerUpdateSpatialWeightsKernel.setArg(index++, _layerDescs[l]._spatialLambda);

		cs.getQueue().enqueueNDRangeKernel(_layerUpdateSpatialWeightsKernel, cl::NullRange, cl::NDRange(_layerDescs[l]._spatialWidth, _layerDescs[l]._spatialHeight));

		// Temporal reconstruction and weight update
		
		Float4 alphas;
		alphas._x = _layerDescs[l]._predictiveAlpha;
		alphas._y = _layerDescs[l]._feedBackAlpha;
		alphas._z = _layerDescs[l]._lateralAlpha;

		Float4 momenta;
		momenta._x = _layerDescs[l]._predictiveMomentum;
		momenta._y = _layerDescs[l]._feedBackMomentum;
		momenta._z = _layerDescs[l]._lateralMomentum;

		index = 0;

		if (l == _layers.size() - 1) {
			_layerUpdateTemporalWeightsLastKernel.setArg(index++, _layers[l]._hiddenStatesSpatial);
			_layerUpdateTemporalWeightsLastKernel.setArg(index++, _layers[l]._hiddenStatesTemporal);
			_layerUpdateTemporalWeightsLastKernel.setArg(index++, _layers[l]._hiddenStatesTemporalPrev);
			_layerUpdateTemporalWeightsLastKernel.setArg(index++, _layers[l]._spatialReconstruction);
			_layerUpdateTemporalWeightsLastKernel.setArg(index++, _layers[l]._temporalReconstruction);
			_layerUpdateTemporalWeightsLastKernel.setArg(index++, _layers[l]._predictiveWeightsPrev);
			_layerUpdateTemporalWeightsLastKernel.setArg(index++, _layers[l]._lateralWeightsPrev);
			_layerUpdateTemporalWeightsLastKernel.setArg(index++, _layers[l]._predictiveWeights);
			_layerUpdateTemporalWeightsLastKernel.setArg(index++, _layers[l]._lateralWeights);
			_layerUpdateTemporalWeightsLastKernel.setArg(index++, layerSizeTemporal);
			_layerUpdateTemporalWeightsLastKernel.setArg(index++, layerSizeTemporalMinusOne);
			_layerUpdateTemporalWeightsLastKernel.setArg(index++, layerSizeTemporalMinusOneInv);
			_layerUpdateTemporalWeightsLastKernel.setArg(index++, layerSizeSpatial);
			_layerUpdateTemporalWeightsLastKernel.setArg(index++, layerSizeSpatialMinusOne);
			_layerUpdateTemporalWeightsLastKernel.setArg(index++, layerSizeSpatialMinusOneInv);
			_layerUpdateTemporalWeightsLastKernel.setArg(index++, _layerDescs[l]._predictiveRadius);
			_layerUpdateTemporalWeightsLastKernel.setArg(index++, _layerDescs[l]._lateralConnectionRadius);
			_layerUpdateTemporalWeightsLastKernel.setArg(index++, _layerDescs[l]._temporalSparsity);
			_layerUpdateTemporalWeightsLastKernel.setArg(index++, _layerDescs[l]._temporalInhibitionRadius);
			_layerUpdateTemporalWeightsLastKernel.setArg(index++, alphas);
			_layerUpdateTemporalWeightsLastKernel.setArg(index++, momenta);
			_layerUpdateTemporalWeightsLastKernel.setArg(index++, _layerDescs[l]._temporalLambda);

			cs.getQueue().enqueueNDRangeKernel(_layerUpdateTemporalWeightsLastKernel, cl::NullRange, cl::NDRange(_layerDescs[l]._temporalWidth, _layerDescs[l]._temporalHeight));
		}
		else {
			_layerUpdateTemporalWeightsKernel.setArg(index++, _layers[l]._hiddenStatesSpatial);
			_layerUpdateTemporalWeightsKernel.setArg(index++, _layers[l]._hiddenStatesTemporal);
			_layerUpdateTemporalWeightsKernel.setArg(index++, _layers[l]._hiddenStatesTemporalPrev);
			_layerUpdateTemporalWeightsKernel.setArg(index++, _layers[l + 1]._hiddenStatesTemporal);
			_layerUpdateTemporalWeightsKernel.setArg(index++, _layers[l]._spatialReconstruction);
			_layerUpdateTemporalWeightsKernel.setArg(index++, _layers[l]._temporalReconstruction);
			_layerUpdateTemporalWeightsKernel.setArg(index++, _layers[l]._nextTemporalReconstruction);
			_layerUpdateTemporalWeightsKernel.setArg(index++, _layers[l]._predictiveWeightsPrev);
			_layerUpdateTemporalWeightsKernel.setArg(index++, _layers[l]._lateralWeightsPrev);
			_layerUpdateTemporalWeightsKernel.setArg(index++, _layers[l]._feedBackWeightsPrev);
			_layerUpdateTemporalWeightsKernel.setArg(index++, _layers[l]._predictiveWeights);
			_layerUpdateTemporalWeightsKernel.setArg(index++, _layers[l]._lateralWeights);
			_layerUpdateTemporalWeightsKernel.setArg(index++, _layers[l]._feedBackWeights);
			_layerUpdateTemporalWeightsKernel.setArg(index++, layerSizeTemporal);
			_layerUpdateTemporalWeightsKernel.setArg(index++, layerSizeTemporalMinusOne);
			_layerUpdateTemporalWeightsKernel.setArg(index++, layerSizeTemporalMinusOneInv);
			_layerUpdateTemporalWeightsKernel.setArg(index++, layerSizeSpatial);
			_layerUpdateTemporalWeightsKernel.setArg(index++, layerSizeSpatialMinusOne);
			_layerUpdateTemporalWeightsKernel.setArg(index++, layerSizeSpatialMinusOneInv);
			_layerUpdateTemporalWeightsKernel.setArg(index++, nextTemporalSize);
			_layerUpdateTemporalWeightsKernel.setArg(index++, nextTemporalSizeMinusOne);
			_layerUpdateTemporalWeightsKernel.setArg(index++, _layerDescs[l]._predictiveRadius);
			_layerUpdateTemporalWeightsKernel.setArg(index++, _layerDescs[l]._lateralConnectionRadius);
			_layerUpdateTemporalWeightsKernel.setArg(index++, _layerDescs[l]._feedBackConnectionRadius);
			_layerUpdateTemporalWeightsKernel.setArg(index++, _layerDescs[l]._temporalSparsity);
			_layerUpdateTemporalWeightsKernel.setArg(index++, _layerDescs[l]._temporalInhibitionRadius);
			_layerUpdateTemporalWeightsKernel.setArg(index++, alphas);
			_layerUpdateTemporalWeightsKernel.setArg(index++, momenta);
			_layerUpdateTemporalWeightsKernel.setArg(index++, _layerDescs[l]._temporalLambda);

			cs.getQueue().enqueueNDRangeKernel(_layerUpdateTemporalWeightsKernel, cl::NullRange, cl::NDRange(_layerDescs[l]._temporalWidth, _layerDescs[l]._temporalHeight));
		}

		index = 0;

		_layerSpatialPredictiveReconstructionWeightUpdateKernel.setArg(index++, _layers[l]._predictedSpatialPrev);
		_layerSpatialPredictiveReconstructionWeightUpdateKernel.setArg(index++, _layers[l]._hiddenStatesSpatial);
		_layerSpatialPredictiveReconstructionWeightUpdateKernel.setArg(index++, _layers[l]._hiddenStatesTemporalPrev);
		_layerSpatialPredictiveReconstructionWeightUpdateKernel.setArg(index++, _layers[l]._spatialPredictiveReconstructionWeightsPrev);
		_layerSpatialPredictiveReconstructionWeightUpdateKernel.setArg(index++, _layers[l]._spatialPredictiveReconstructionWeights);
		_layerSpatialPredictiveReconstructionWeightUpdateKernel.setArg(index++, _layerDescs[l]._reconstructionRadius);
		_layerSpatialPredictiveReconstructionWeightUpdateKernel.setArg(index++, layerSizeSpatialMinusOne);
		_layerSpatialPredictiveReconstructionWeightUpdateKernel.setArg(index++, layerSizeSpatialMinusOneInv);
		_layerSpatialPredictiveReconstructionWeightUpdateKernel.setArg(index++, layerSizeTemporal);
		_layerSpatialPredictiveReconstructionWeightUpdateKernel.setArg(index++, layerSizeTemporalMinusOne);
		_layerSpatialPredictiveReconstructionWeightUpdateKernel.setArg(index++, layerSizeTemporalMinusOneInv);
		_layerSpatialPredictiveReconstructionWeightUpdateKernel.setArg(index++, _layerDescs[l]._reconstructionAlpha);
		_layerSpatialPredictiveReconstructionWeightUpdateKernel.setArg(index++, _layerDescs[l]._reconstructionMomentum);

		cs.getQueue().enqueueNDRangeKernel(_layerSpatialPredictiveReconstructionWeightUpdateKernel, cl::NullRange, cl::NDRange(_layerDescs[l]._spatialWidth, _layerDescs[l]._spatialHeight));

		pPrevLayer = &_layers[l]._hiddenStatesSpatial;
		prevWidth = _layerDescs[l]._spatialWidth;
		prevHeight = _layerDescs[l]._spatialHeight;
	}
}

void HTFE::stepEnd() {
	// ------------------------------------------------------------------------------
	// ---------------------------------- Step End ----------------------------------
	// ------------------------------------------------------------------------------

	for (int l = 0; l < _layers.size(); l++) {
		cl::Image2D temp2D;

		std::swap(_layers[l]._hiddenStatesSpatial, _layers[l]._hiddenStatesSpatialPrev);

		temp2D = _layers[l]._hiddenStatesTemporalPrevPrev;
		_layers[l]._hiddenStatesTemporalPrevPrev = _layers[l]._hiddenStatesTemporalPrev;
		_layers[l]._hiddenStatesTemporalPrev = _layers[l]._hiddenStatesTemporal;
		_layers[l]._hiddenStatesTemporal = temp2D;

		std::swap(_layers[l]._spatialWeights, _layers[l]._spatialWeightsPrev);
		std::swap(_layers[l]._spatialPredictiveReconstructionWeights, _layers[l]._spatialPredictiveReconstructionWeightsPrev);
		std::swap(_layers[l]._predictiveWeights, _layers[l]._predictiveWeightsPrev);
		std::swap(_layers[l]._lateralWeights, _layers[l]._lateralWeightsPrev);
		std::swap(_layers[l]._feedBackWeights, _layers[l]._feedBackWeightsPrev);
		std::swap(_layers[l]._predictedSpatial, _layers[l]._predictedSpatialPrev);

		std::swap(_layers[l]._spatialReconstruction, _layers[l]._spatialReconstructionPrev);
		std::swap(_layers[l]._temporalReconstruction, _layers[l]._temporalReconstructionPrev);
		std::swap(_layers[l]._nextTemporalReconstruction, _layers[l]._nextTemporalReconstructionPrev);
	}

	std::swap(_inputImage, _inputImagePrev);
}

void HTFE::gaussianBlur(sys::ComputeSystem &cs, cl::Image2D &source, cl::Image2D &ping, cl::Image2D &pong, int imageSizeX, int imageSizeY, int passes, float kernelWidth) {
	Float2 imageSizeInv;
	imageSizeInv._x = 1.0f / imageSizeX;
	imageSizeInv._y = 1.0f / imageSizeY;

	// Blur source to ping
	_gaussianBlurXKernel.setArg(0, source);
	_gaussianBlurXKernel.setArg(1, ping);
	_gaussianBlurXKernel.setArg(2, imageSizeInv);
	_gaussianBlurXKernel.setArg(3, kernelWidth * imageSizeInv._x);

	cs.getQueue().enqueueNDRangeKernel(_gaussianBlurXKernel, cl::NullRange, cl::NDRange(imageSizeX, imageSizeY));

	for (int p = 0; p < passes - 1; p++) {
		_gaussianBlurYKernel.setArg(0, ping);
		_gaussianBlurYKernel.setArg(1, pong);
		_gaussianBlurYKernel.setArg(2, imageSizeInv);
		_gaussianBlurYKernel.setArg(3, kernelWidth * imageSizeInv._y);

		cs.getQueue().enqueueNDRangeKernel(_gaussianBlurYKernel, cl::NullRange, cl::NDRange(imageSizeX, imageSizeY));

		_gaussianBlurXKernel.setArg(0, pong);
		_gaussianBlurXKernel.setArg(1, ping);
		_gaussianBlurXKernel.setArg(2, imageSizeInv);
		_gaussianBlurXKernel.setArg(3, kernelWidth * imageSizeInv._x);

		cs.getQueue().enqueueNDRangeKernel(_gaussianBlurXKernel, cl::NullRange, cl::NDRange(imageSizeX, imageSizeY));
	}

	_gaussianBlurYKernel.setArg(0, ping);
	_gaussianBlurYKernel.setArg(1, pong);
	_gaussianBlurYKernel.setArg(2, imageSizeInv);
	_gaussianBlurYKernel.setArg(3, kernelWidth * imageSizeInv._y);

	cs.getQueue().enqueueNDRangeKernel(_gaussianBlurYKernel, cl::NullRange, cl::NDRange(imageSizeX, imageSizeY));
}