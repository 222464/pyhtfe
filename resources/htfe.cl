constant sampler_t normalizedClampedNearestSampler = CLK_NORMALIZED_COORDS_TRUE |
CLK_ADDRESS_CLAMP |
CLK_FILTER_NEAREST;

constant sampler_t normalizedClampedToEdgeNearestSampler = CLK_NORMALIZED_COORDS_TRUE |
CLK_ADDRESS_CLAMP_TO_EDGE |
CLK_FILTER_NEAREST;

constant sampler_t unnormalizedClampedNearestSampler = CLK_NORMALIZED_COORDS_FALSE |
CLK_ADDRESS_CLAMP |
CLK_FILTER_NEAREST;

constant sampler_t defaultNormalizedSampler = CLK_NORMALIZED_COORDS_TRUE |
CLK_ADDRESS_CLAMP_TO_EDGE |
CLK_FILTER_NEAREST;

constant sampler_t defaultUnnormalizedSampler = CLK_NORMALIZED_COORDS_FALSE |
CLK_ADDRESS_CLAMP_TO_EDGE |
CLK_FILTER_NEAREST;

float randFloat(uint2* state) {
	const float invMaxInt = 1.0f / 4294967296.0f;
	uint x = (*state).x * 17 + (*state).y * 13123;
	(*state).x = (x << 13) ^ x;
	(*state).y ^= (x << 7);

	uint tmp = x * (x * x * 15731 + 74323) + 871483;

	return convert_float(tmp) * invMaxInt;
}

float randNormal(uint2* state) {
	float u1 = randFloat(state);
	float u2 = randFloat(state);

	return sqrt(-2.0f * log(u1)) * cos(6.28318f * u2);
}

float sigmoid(float x) {
	return 1.0f / (1.0f + exp(-x));
}

float boostFunction(float trace, float threshold) {
	return fmin(1.0f, fmax(0.0f, threshold - trace) / threshold);
}

void kernel initializeLayerHiddenSpatial(
	write_only image3d_t spatialWeights,
	write_only image3d_t spatialPredictiveReconstructionWeights,
	write_only image2d_t hiddenStatesSpatial,
	int spatialSize, int reconstructionSize,
	uint2 seed, float spatialSparsity, float minWeight, float maxWeight)
{
	uint2 seedValue = seed + (uint2)(get_global_id(0) * 29 + 12, get_global_id(1) * 16 + 23) * 36;

	int2 hiddenPosition = (int2)(get_global_id(0), get_global_id(1));

	write_imagef(hiddenStatesSpatial, hiddenPosition, (float4)(0.0f, 0.0f, spatialSparsity, 0.0f));

	for (int wi = 0; wi < spatialSize; wi++) {
		int4 weightPosition = (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0);

		float spatialWeight = randFloat(&seedValue) * (maxWeight - minWeight) + minWeight;

		write_imagef(spatialWeights, weightPosition, (float4)(spatialWeight, 0.0f, 0.0f, 0.0f));
	}

	for (int wi = 0; wi < reconstructionSize; wi++) {
		int4 weightPosition = (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0);

		float reconstructionWeight = randFloat(&seedValue) * (maxWeight - minWeight) + minWeight;

		write_imagef(spatialPredictiveReconstructionWeights, weightPosition, (float4)(reconstructionWeight, 0.0f, 0.0f, 0.0f));
	}
}

void kernel initializeLayerHiddenTemporal(
	write_only image3d_t predictiveWeights,
	write_only image3d_t lateralWeights,
	write_only image3d_t feedBackWeights,
	write_only image2d_t hiddenStatesTemporal,
	int predictiveSize, int lateralSize, int feedBackSize,
	uint2 seed, float temporalSparsity, float lateralScalar, float feedBackScalar, float minWeight, float maxWeight)
{
	uint2 seedValue = seed + (uint2)(get_global_id(0) * 29 + 12, get_global_id(1) * 16 + 23) * 36;

	int2 hiddenPosition = (int2)(get_global_id(0), get_global_id(1));

	write_imagef(hiddenStatesTemporal, hiddenPosition, (float4)(0.0f, 0.0f, temporalSparsity, 0.0f));

	for (int wi = 0; wi < predictiveSize; wi++) {
		int4 weightPosition = (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0);

		float predictiveWeight = randFloat(&seedValue) * (maxWeight - minWeight) + minWeight;

		write_imagef(predictiveWeights, weightPosition, (float4)(predictiveWeight, 0.0f, 0.0f, 0.0f));
	}

	for (int wi = 0; wi < lateralSize; wi++) {
		int4 weightPosition = (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0);

		float lateralWeight = lateralScalar * (randFloat(&seedValue) * (maxWeight - minWeight) + minWeight);

		write_imagef(lateralWeights, weightPosition, (float4)(lateralWeight, 0.0f, 0.0f, 0.0f));
	}

	for (int wi = 0; wi < feedBackSize; wi++) {
		int4 weightPosition = (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0);

		float feedBackWeight = feedBackScalar * (randFloat(&seedValue) * (maxWeight - minWeight) + minWeight);

		write_imagef(feedBackWeights, weightPosition, (float4)(feedBackWeight, 0.0f, 0.0f, 0.0f));
	}
}

void kernel layerInhibit(read_only image2d_t activations, write_only image2d_t states,
	int2 layerSize, int inhibitionRadius, float localActivity, float minDerivative)
{
	int2 hiddenPosition = (int2)(get_global_id(0), get_global_id(1));

	float thisActivation = read_imagef(activations, hiddenPosition).x;

	float numHigher = 0.0f;

	for (int dx = -inhibitionRadius; dx <= inhibitionRadius; dx++)
		for (int dy = -inhibitionRadius; dy <= inhibitionRadius; dy++) {
			int2 layerPosition = (int2)(hiddenPosition.x + dx, hiddenPosition.y + dy);

			if (layerPosition.x >= 0 && layerPosition.x < layerSize.x && layerPosition.y >= 0 && layerPosition.y < layerSize.y) {
				float activation = read_imagef(activations, layerPosition).x;

				numHigher += activation >= thisActivation ? 1.0f : 0.0f;
			}
		}

	float newState = numHigher < localActivity ? 1.0f : 0.0f;

	float newDeriv = numHigher < localActivity ? thisActivation * (1.0f - thisActivation) : minDerivative * thisActivation * (1.0f - thisActivation);

	write_imagef(states, hiddenPosition, (float4)(newState, newDeriv, 0.0f, 0.0f));
}

void kernel layerHiddenStatesSpatialActivate(read_only image2d_t inputs, read_only image3d_t spatialWeights, write_only image2d_t hiddenStatesSpatial,
	int2 layerSize, float2 layerSizeMinusOneInv, int2 inputSize, int2 inputSizeMinusOne, int receptiveFieldRadius, float dutyCycleDecay, float minDerivative, uint2 seed)
{
	uint2 seedValue = seed + (uint2)(get_global_id(0) * 12 + 62, get_global_id(1) * 8 + 2) * 4;

	int2 hiddenPosition = (int2)(get_global_id(0), get_global_id(1));

	float2 inputCenterPositionNormalized = (float2)(hiddenPosition.x * layerSizeMinusOneInv.x, hiddenPosition.y * layerSizeMinusOneInv.y);
	int2 inputCenterPosition = (int2)(inputCenterPositionNormalized.x * inputSizeMinusOne.x, inputCenterPositionNormalized.y * inputSizeMinusOne.y);

	float sum = 0.0f;

	int wi = 0;

	for (int dx = -receptiveFieldRadius; dx <= receptiveFieldRadius; dx++)
		for (int dy = -receptiveFieldRadius; dy <= receptiveFieldRadius; dy++) {
			int2 inputPosition = (int2)(inputCenterPosition.x + dx, inputCenterPosition.y + dy);

			if (inputPosition.x >= 0 && inputPosition.x < inputSize.x && inputPosition.y >= 0 && inputPosition.y < inputSize.y) {
				float input = read_imagef(inputs, inputPosition).x;

				float weight = read_imagef(spatialWeights, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0)).x;

				sum += weight * input;
			}

			wi++;
		}

	// Bias
	float bias = read_imagef(spatialWeights, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0)).x;

	sum += bias;

	write_imagef(hiddenStatesSpatial, hiddenPosition, (float4)(sigmoid(sum), 0.0f, 0.0f, 0.0f));
}

void kernel layerHiddenStatesTemporalActivate(read_only image2d_t hiddenStatesSpatial, read_only image2d_t hiddenStatesTemporalPrev, read_only image2d_t nextLayerHiddenStatesTemporal,
	read_only image3d_t predictiveWeights, read_only image3d_t feedBackWeights, read_only image3d_t lateralWeights, write_only image2d_t hiddenStatesTemporal,
	int2 layerSize, float2 layerSizeMinusOneInv, int2 inputSize, int2 inputSizeMinusOne, int2 nextSize, int2 nextSizeMinusOne, int predictiveRadius, int feedBackRadius, int lateralConnectionRadius, float dutyCycleDecay, float minDerivative, uint2 seed)
{
	uint2 seedValue = seed + (uint2)(get_global_id(0) * 56 + 2, get_global_id(1) * 6 + 4) * 3;

	int2 hiddenPosition = (int2)(get_global_id(0), get_global_id(1));

	float2 positionNormalized = (float2)(hiddenPosition.x * layerSizeMinusOneInv.x, hiddenPosition.y * layerSizeMinusOneInv.y);
	int2 nextCenterPosition = (int2)(positionNormalized.x * nextSizeMinusOne.x, positionNormalized.y * nextSizeMinusOne.y);
	int2 inputCenterPosition = (int2)(positionNormalized.x * inputSizeMinusOne.x, positionNormalized.y * inputSizeMinusOne.y);

	float sum = 0.0f;

	int wi = 0;

	for (int dx = -predictiveRadius; dx <= predictiveRadius; dx++)
		for (int dy = -predictiveRadius; dy <= predictiveRadius; dy++) {
			int2 layerPosition = (int2)(inputCenterPosition.x + dx, inputCenterPosition.y + dy);

			if (layerPosition.x >= 0 && layerPosition.x < inputSize.x && layerPosition.y >= 0 && layerPosition.y < inputSize.y) {
				float state = read_imagef(hiddenStatesSpatial, layerPosition).x;

				float weight = read_imagef(predictiveWeights, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0)).x;

				sum += weight * state;
			}

			wi++;
		}

	// Bias
	float bias = read_imagef(predictiveWeights, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0)).x;

	sum += bias;

	wi = 0;

	for (int dx = -feedBackRadius; dx <= feedBackRadius; dx++)
		for (int dy = -feedBackRadius; dy <= feedBackRadius; dy++) {
			int2 nextPosition = (int2)(nextCenterPosition.x + dx, nextCenterPosition.y + dy);

			if (nextPosition.x >= 0 && nextPosition.x < nextSize.x && nextPosition.y >= 0 && nextPosition.y < nextSize.y) {
				float next = read_imagef(nextLayerHiddenStatesTemporal, nextPosition).x;

				float weight = read_imagef(feedBackWeights, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0)).x;

				sum += weight * next;
			}

			wi++;
		}

	wi = 0;

	for (int dx = -lateralConnectionRadius; dx <= lateralConnectionRadius; dx++)
		for (int dy = -lateralConnectionRadius; dy <= lateralConnectionRadius; dy++) {
			int2 layerPosition = (int2)(hiddenPosition.x + dx, hiddenPosition.y + dy);

			if (layerPosition.x >= 0 && layerPosition.x < layerSize.x && layerPosition.y >= 0 && layerPosition.y < layerSize.y) {
				float state = read_imagef(hiddenStatesTemporalPrev, layerPosition).x;

				float weight = read_imagef(lateralWeights, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0)).x;

				sum += weight * state;
			}

			wi++;
		}

	write_imagef(hiddenStatesTemporal, hiddenPosition, (float4)(sigmoid(sum), 0.0f, 0.0f, 0.0f));
}

void kernel layerHiddenStatesTemporalActivateLast(read_only image2d_t hiddenStatesSpatial, read_only image2d_t hiddenStatesTemporalPrev,
	read_only image3d_t predictiveWeights, read_only image3d_t lateralWeights, write_only image2d_t hiddenStatesTemporal,
	int2 layerSize, float2 layerSizeMinusOneInv, int2 inputSize, int2 inputSizeMinusOne, int predictiveRadius, int lateralConnectionRadius, float dutyCycleDecay, float minDerivative, uint2 seed)
{
	uint2 seedValue = seed + (uint2)(get_global_id(0) * 56 + 2, get_global_id(1) * 6 + 4) * 3;

	int2 hiddenPosition = (int2)(get_global_id(0), get_global_id(1));
	float2 positionNormalized = (float2)(hiddenPosition.x * layerSizeMinusOneInv.x, hiddenPosition.y * layerSizeMinusOneInv.y);
	int2 inputCenterPosition = (int2)(positionNormalized.x * inputSizeMinusOne.x, positionNormalized.y * inputSizeMinusOne.y);

	float sum = 0.0f;

	int wi = 0;

	for (int dx = -predictiveRadius; dx <= predictiveRadius; dx++)
		for (int dy = -predictiveRadius; dy <= predictiveRadius; dy++) {
			int2 layerPosition = (int2)(inputCenterPosition.x + dx, inputCenterPosition.y + dy);

			if (layerPosition.x >= 0 && layerPosition.x < inputSize.x && layerPosition.y >= 0 && layerPosition.y < inputSize.y) {
				float state = read_imagef(hiddenStatesSpatial, layerPosition).x;

				float weight = read_imagef(predictiveWeights, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0)).x;

				sum += weight * state;
			}

			wi++;
		}

	// Bias
	float bias = read_imagef(predictiveWeights, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0)).x;

	sum += bias;

	wi = 0;

	for (int dx = -lateralConnectionRadius; dx <= lateralConnectionRadius; dx++)
		for (int dy = -lateralConnectionRadius; dy <= lateralConnectionRadius; dy++) {
			int2 layerPosition = (int2)(hiddenPosition.x + dx, hiddenPosition.y + dy);

			if (layerPosition.x >= 0 && layerPosition.x < layerSize.x && layerPosition.y >= 0 && layerPosition.y < layerSize.y) {
				float state = read_imagef(hiddenStatesTemporalPrev, layerPosition).x;

				float weight = read_imagef(lateralWeights, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0)).x;

				sum += weight * state;
			}

			wi++;
		}

	write_imagef(hiddenStatesTemporal, hiddenPosition, (float4)(sigmoid(sum), 0.0f, 0.0f, 0.0f));
}

void kernel layerInputReconstruct(read_only image2d_t hiddenStates, read_only image3d_t feedForwardWeights, write_only image2d_t inputReconstruction,
	int receptiveRadius, int2 reverseReceptiveRadius, int2 inputSizeMinusOne, float2 inputSizeMinusOneInv, int2 layerSize, int2 layerSizeMinusOne, float2 layerSizeMinusOneInv, uint2 seed)
{
	uint2 seedValue = seed + (uint2)(get_global_id(0) * 23 + 9, get_global_id(1) * 5 + 2) * 2;

	int2 visiblePosition = (int2)(get_global_id(0), get_global_id(1));
	float2 layerPositionNormalized = (float2)(visiblePosition.x * inputSizeMinusOneInv.x, visiblePosition.y * inputSizeMinusOneInv.y);
	int2 layerPositionCenter = (int2)(layerPositionNormalized.x * layerSizeMinusOne.x, layerPositionNormalized.y * layerSizeMinusOne.y);

	float sum = 0.0f;

	for (int dx = -reverseReceptiveRadius.x; dx <= reverseReceptiveRadius.x; dx++)
		for (int dy = -reverseReceptiveRadius.y; dy <= reverseReceptiveRadius.y; dy++) {
			int2 layerPosition = (int2)(layerPositionCenter.x + dx, layerPositionCenter.y + dy);

			if (layerPosition.x >= 0 && layerPosition.x < layerSize.x && layerPosition.y >= 0 && layerPosition.y < layerSize.y) {
				// Next layer node's receptive field
				int2 fieldCenter = (int2)(layerPosition.x * layerSizeMinusOneInv.x * inputSizeMinusOne.x, layerPosition.y * layerSizeMinusOneInv.y * inputSizeMinusOne.y);

				int2 fieldLowerBounds = fieldCenter - (int2)(receptiveRadius);
				int2 fieldUpperBounds = fieldCenter + (int2)(receptiveRadius);

				// Check for containment
				if (visiblePosition.x >= fieldLowerBounds.x && visiblePosition.x <= fieldUpperBounds.x && visiblePosition.y >= fieldLowerBounds.y && visiblePosition.y <= fieldUpperBounds.y) {
					int rdx = visiblePosition.x - fieldLowerBounds.x;
					int rdy = visiblePosition.y - fieldLowerBounds.y;

					float input = read_imagef(hiddenStates, layerPosition).x;
					
					int weightIndex = rdy + rdx * (receptiveRadius * 2 + 1);

					float weight = read_imagef(feedForwardWeights, (int4)(layerPosition.x, layerPosition.y, weightIndex, 0)).x;

					sum += input * weight;
				}
			}
		}

	float recon = sum > 0.0f ? 1.0f : 0.0f;

	write_imagef(inputReconstruction, visiblePosition, (float4)(recon, 0.0f, 0.0f, 0.0f));
}

void kernel layerInputReconstructLinear(read_only image2d_t hiddenStates, read_only image3d_t feedForwardWeights, write_only image2d_t inputReconstruction,
	int receptiveRadius, int2 reverseReceptiveRadius, int2 inputSizeMinusOne, float2 inputSizeMinusOneInv, int2 layerSize, int2 layerSizeMinusOne, float2 layerSizeMinusOneInv, uint2 seed, float noise)
{
	uint2 seedValue = seed + (uint2)(get_global_id(0) * 23 + 9, get_global_id(1) * 5 + 2) * 2;

	int2 visiblePosition = (int2)(get_global_id(0), get_global_id(1));
	float2 layerPositionNormalized = (float2)(visiblePosition.x * inputSizeMinusOneInv.x, visiblePosition.y * inputSizeMinusOneInv.y);
	int2 layerPositionCenter = (int2)(layerPositionNormalized.x * layerSizeMinusOne.x, layerPositionNormalized.y * layerSizeMinusOne.y);

	float sum = 0.0f;

	for (int dx = -reverseReceptiveRadius.x; dx <= reverseReceptiveRadius.x; dx++)
		for (int dy = -reverseReceptiveRadius.y; dy <= reverseReceptiveRadius.y; dy++) {
			int2 layerPosition = (int2)(layerPositionCenter.x + dx, layerPositionCenter.y + dy);

			if (layerPosition.x >= 0 && layerPosition.x < layerSize.x && layerPosition.y >= 0 && layerPosition.y < layerSize.y) {
				// Next layer node's receptive field
				int2 fieldCenter = (int2)(layerPosition.x * layerSizeMinusOneInv.x * inputSizeMinusOne.x, layerPosition.y * layerSizeMinusOneInv.y * inputSizeMinusOne.y);

				int2 fieldLowerBounds = fieldCenter - (int2)(receptiveRadius);
				int2 fieldUpperBounds = fieldCenter + (int2)(receptiveRadius);

				// Check for containment
				if (visiblePosition.x >= fieldLowerBounds.x && visiblePosition.x <= fieldUpperBounds.x && visiblePosition.y >= fieldLowerBounds.y && visiblePosition.y <= fieldUpperBounds.y) {
					int rdx = visiblePosition.x - fieldLowerBounds.x;
					int rdy = visiblePosition.y - fieldLowerBounds.y;

					float input = read_imagef(hiddenStates, layerPosition).x;

					int weightIndex = rdy + rdx * (receptiveRadius * 2 + 1);

					float weight = read_imagef(feedForwardWeights, (int4)(layerPosition.x, layerPosition.y, weightIndex, 0)).x;

					sum += input * weight;
				}
			}
		}

	float recon = sum;

	write_imagef(inputReconstruction, visiblePosition, (float4)(recon, 0.0f, 0.0f, 0.0f));
}

void kernel layerSpatialReconstruct(read_only image2d_t hiddenStates, read_only image3d_t predictiveWeights, write_only image2d_t spatialReconstruction,
	int predictiveRadius, int2 reversePredictiveRadius, int2 inputSize, int2 inputSizeMinusOne, float2 inputSizeMinusOneInv, int2 layerSize, int2 layerSizeMinusOne, float2 layerSizeMinusOneInv, uint2 seed)
{
	uint2 seedValue = seed + (uint2)(get_global_id(0) * 2 + 11, get_global_id(1) * 5 + 7) * 5;

	int2 hiddenPosition = (int2)(get_global_id(0), get_global_id(1));

	float2 positionNormalized = (float2)(hiddenPosition.x * inputSizeMinusOneInv.x, hiddenPosition.y * inputSizeMinusOneInv.y);
	int2 layerCenterPosition = (int2)(positionNormalized.x * layerSizeMinusOne.x, positionNormalized.y * layerSizeMinusOne.y);

	float sum = 0.0f;

	for (int dx = -reversePredictiveRadius.x; dx <= reversePredictiveRadius.x; dx++)
		for (int dy = -reversePredictiveRadius.y; dy <= reversePredictiveRadius.y; dy++) {
			int2 layerPosition = (int2)(layerCenterPosition.x + dx, layerCenterPosition.y + dy);

			if (layerPosition.x >= 0 && layerPosition.x < layerSize.x && layerPosition.y >= 0 && layerPosition.y < layerSize.y) {
				// Next layer node's receptive field
				int2 fieldCenter = (int2)(layerPosition.x * layerSizeMinusOneInv.x * inputSizeMinusOne.x, layerPosition.y * layerSizeMinusOneInv.y * inputSizeMinusOne.y);

				int2 fieldLowerBounds = fieldCenter - (int2)(predictiveRadius);
				int2 fieldUpperBounds = fieldCenter + (int2)(predictiveRadius);

				// Check for containment
				if (hiddenPosition.x >= fieldLowerBounds.x && hiddenPosition.x <= fieldUpperBounds.x && hiddenPosition.y >= fieldLowerBounds.y && hiddenPosition.y <= fieldUpperBounds.y) {
					int rdx = hiddenPosition.x - fieldLowerBounds.x;
					int rdy = hiddenPosition.y - fieldLowerBounds.y;

					float input = read_imagef(hiddenStates, layerPosition).x;

					int weightIndex = rdy + rdx * (predictiveRadius * 2 + 1);

					float weight = read_imagef(predictiveWeights, (int4)(layerPosition.x, layerPosition.y, weightIndex, 0)).x;

					sum += input * weight;
				}
			}
		}

	float recon = sum > 0.0f ? 1.0f : 0.0f;

	write_imagef(spatialReconstruction, hiddenPosition, (float4)(recon, 0.0f, 0.0f, 0.0f));
}

void kernel layerTemporalReconstruct(read_only image2d_t hiddenStates, read_only image3d_t lateralWeights, write_only image2d_t temporalReconstruction,
	int lateralRadius, int2 layerSize, uint2 seed)
{
	uint2 seedValue = seed + (uint2)(get_global_id(0) * 32 + 2, get_global_id(1) * 6 + 1) * 6;

	int2 hiddenPosition = (int2)(get_global_id(0), get_global_id(1));

	float sum = 0.0f;

	for (int dx = -lateralRadius; dx <= lateralRadius; dx++)
		for (int dy = -lateralRadius; dy <= lateralRadius; dy++) {
			int2 layerPosition = (int2)(hiddenPosition.x + dx, hiddenPosition.y + dy);

			if (layerPosition.x >= 0 && layerPosition.x < layerSize.x && layerPosition.y >= 0 && layerPosition.y < layerSize.y) {
				float input = read_imagef(hiddenStates, layerPosition).x;

				int weightIndex = lateralRadius - dy + (lateralRadius - dx) * (lateralRadius * 2 + 1);

				float weight = read_imagef(lateralWeights, (int4)(layerPosition.x, layerPosition.y, weightIndex, 0)).x;

				sum += input * weight;
			}
		}

	float recon = sum > 0.0f ? 1.0f : 0.0f;

	write_imagef(temporalReconstruction, hiddenPosition, (float4)(recon, 0.0f, 0.0f, 0.0f));
}

void kernel layerNextTemporalReconstruct(read_only image2d_t hiddenStatesTemporal, read_only image3d_t feedBackWeights, write_only image2d_t nextTemporalReconstruction,
	int feedBackRadius, int2 reverseFeedBackRadius, int2 nextSizeMinusOne, float2 nextSizeMinusOneInv, int2 layerSize, int2 layerSizeMinusOne, float2 layerSizeMinusOneInv, uint2 seed)
{
	uint2 seedValue = seed + (uint2)(get_global_id(0) * 4 + 23, get_global_id(1) * 2 + 44) * 3;

	int2 hiddenPosition = (int2)(get_global_id(0), get_global_id(1));

	float2 positionNormalized = (float2)(hiddenPosition.x * nextSizeMinusOneInv.x, hiddenPosition.y * nextSizeMinusOneInv.y);
	int2 layerCenterPosition = (int2)(positionNormalized.x * layerSizeMinusOne.x, positionNormalized.y * layerSizeMinusOne.y);

	float sum = 0.0f;

	for (int dx = -reverseFeedBackRadius.x; dx <= reverseFeedBackRadius.x; dx++)
		for (int dy = -reverseFeedBackRadius.y; dy <= reverseFeedBackRadius.y; dy++) {
			int2 prevPosition = (int2)(layerCenterPosition.x + dx, layerCenterPosition.y + dy);

			if (prevPosition.x >= 0 && prevPosition.x < layerSize.x && prevPosition.y >= 0 && prevPosition.y < layerSize.y) {
				// Next layer node's receptive field
				int2 fieldCenter = (int2)(prevPosition.x * layerSizeMinusOneInv.x * nextSizeMinusOne.x, prevPosition.y * layerSizeMinusOneInv.y * nextSizeMinusOne.y);

				int2 fieldLowerBounds = fieldCenter - (int2)(feedBackRadius);
				int2 fieldUpperBounds = fieldCenter + (int2)(feedBackRadius);

				// Check for containment
				if (hiddenPosition.x >= fieldLowerBounds.x && hiddenPosition.x <= fieldUpperBounds.x && hiddenPosition.y >= fieldLowerBounds.y && hiddenPosition.y <= fieldUpperBounds.y) {
					int rdx = hiddenPosition.x - fieldLowerBounds.x;
					int rdy = hiddenPosition.y - fieldLowerBounds.y;

					float input = read_imagef(hiddenStatesTemporal, prevPosition).x;

					int weightIndex = rdy + rdx * (feedBackRadius * 2 + 1);

					float weight = read_imagef(feedBackWeights, (int4)(prevPosition.x, prevPosition.y, weightIndex, 0)).x;

					sum += input * weight;
				}
			}
		}

	float recon = sum > 0.0f ? 1.0f : 0.0f;

	write_imagef(nextTemporalReconstruction, hiddenPosition, (float4)(recon, 0.0f, 0.0f, 0.0f));
}

void kernel layerUpdateSpatialWeights(read_only image2d_t inputs, read_only image2d_t spatialReconstruction, read_only image2d_t hiddenStates, read_only image3d_t feedForwardWeightsPrev, write_only image3d_t feedForwardWeights,
	int2 layerSize, float2 layerSizeMinusOneInv, int2 inputSize, int2 inputSizeMinusOne, int receptiveFieldRadius, int inhibitionRadius, float sparsity, float alpha, float momentum, float lambda)
{
	int2 hiddenPosition = (int2)(get_global_id(0), get_global_id(1));

	float2 inputCenterPositionNormalized = (float2)(hiddenPosition.x * layerSizeMinusOneInv.x, hiddenPosition.y * layerSizeMinusOneInv.y);
	int2 inputCenterPosition = (int2)(inputCenterPositionNormalized.x * inputSizeMinusOne.x, inputCenterPositionNormalized.y * inputSizeMinusOne.y);

	float sum = 0.0f;

	int wi = 0;

	for (int dx = -receptiveFieldRadius; dx <= receptiveFieldRadius; dx++)
		for (int dy = -receptiveFieldRadius; dy <= receptiveFieldRadius; dy++) {
			int2 inputPosition = (int2)(inputCenterPosition.x + dx, inputCenterPosition.y + dy);

			if (inputPosition.x >= 0 && inputPosition.x < inputSize.x && inputPosition.y >= 0 && inputPosition.y < inputSize.y) {
				float input = read_imagef(inputs, inputPosition).x;
				float recon = read_imagef(spatialReconstruction, inputPosition).x;

				float weight = read_imagef(feedForwardWeightsPrev, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0)).x;

				sum += (input - recon) * weight;
			}

			wi++;
		}

	// Bias
	//float bias = read_imagef(feedForwardWeightsPrev, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0)).x;

	//sum += bias;

	float2 hiddenState = read_imagef(hiddenStates, hiddenPosition).xy;

	sum *= hiddenState.y;

	wi = 0;

	for (int dx = -receptiveFieldRadius; dx <= receptiveFieldRadius; dx++)
		for (int dy = -receptiveFieldRadius; dy <= receptiveFieldRadius; dy++) {
			int2 inputPosition = (int2)(inputCenterPosition.x + dx, inputCenterPosition.y + dy);

			if (inputPosition.x >= 0 && inputPosition.x < inputSize.x && inputPosition.y >= 0 && inputPosition.y < inputSize.y) {
				float input = read_imagef(inputs, inputPosition).x;
				float recon = read_imagef(spatialReconstruction, inputPosition).x;

				// CD
				float eligibility = 0.5f * (sum * input + (input - recon) * hiddenState.x);

				float2 prevWeight = read_imagef(feedForwardWeightsPrev, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0)).xy;

				float newWeight = prevWeight.x + alpha * eligibility + momentum * prevWeight.y;
				float newDelta = newWeight - prevWeight.x;

				write_imagef(feedForwardWeights, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0), (float4)(newWeight, newDelta, 0.0f, 0.0f));
			}

			wi++;
		}

	// Bias - CD
	float eligibility = sum;

	float2 prevBias = read_imagef(feedForwardWeightsPrev, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0)).xy;

	float newBias = prevBias.x + alpha * eligibility + momentum * prevBias.y;
	float newDelta = newBias - prevBias.x;

	write_imagef(feedForwardWeights, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0), (float4)(newBias, newDelta, 0.0f, 0.0f));
}

void kernel layerSpatialPredictiveReconstruct(read_only image2d_t hiddenStates, read_only image3d_t reconstructionWeights, write_only image2d_t predictedSpatial,
	int reconstructionReceptiveRadius, int2 inputSizeMinusOne, float2 inputSizeMinusOneInv, int2 layerSize, int2 layerSizeMinusOne, float2 layerSizeMinusOneInv)
{
	int2 visiblePosition = (int2)(get_global_id(0), get_global_id(1));
	float2 layerPositionNormalized = (float2)(visiblePosition.x * inputSizeMinusOneInv.x, visiblePosition.y * inputSizeMinusOneInv.y);
	int2 layerPositionCenter = (int2)(layerPositionNormalized.x * layerSizeMinusOne.x, layerPositionNormalized.y * layerSizeMinusOne.y);

	float sum = 0.0f;

	int wi = 0;

	for (int dx = -reconstructionReceptiveRadius; dx <= reconstructionReceptiveRadius; dx++)
		for (int dy = -reconstructionReceptiveRadius; dy <= reconstructionReceptiveRadius; dy++) {
			int2 layerPosition = (int2)(layerPositionCenter.x + dx, layerPositionCenter.y + dy);

			if (layerPosition.x >= 0 && layerPosition.x < layerSize.x && layerPosition.y >= 0 && layerPosition.y < layerSize.y) {
				float source = read_imagef(hiddenStates, layerPosition).x;

				float weight = read_imagef(reconstructionWeights, (int4)(visiblePosition.x, visiblePosition.y, wi, 0)).x;

				sum += source * weight;
			}

			wi++;
		}

	float recon = sum > 0.0f ? 1.0f : 0.0f;

	write_imagef(predictedSpatial, visiblePosition, (float4)(recon, 0.0f, 0.0f, 0.0f));
}

void kernel layerUpdateTemporalWeights(read_only image2d_t hiddenStatesSpatial, read_only image2d_t hiddenStatesTemporal, read_only image2d_t hiddenStatesTemporalPrev, read_only image2d_t hiddenStatesNextTemporal,
	read_only image2d_t spatialReconstruction, read_only image2d_t temporalReconstruction, read_only image2d_t nextTemporalReconstruction,
	read_only image3d_t predictiveWeightsPrev, read_only image3d_t lateralWeightsPrev, read_only image3d_t feedBackWeightsPrev,
	write_only image3d_t predictiveWeights, write_only image3d_t lateralWeights, write_only image3d_t feedBackWeights,
	int2 layerSize, int2 layerSizeMinusOne, float2 layerSizeMinusOneInv, int2 inputSize, int2 inputSizeMinusOne, float2 inputSizeMinusOneInv, int2 nextSize, int2 nextSizeMinusOne,
	int predictiveRadius, int lateralConnectionRadius, int feedBackRadius,
	float sparsity, int inhibitionRadius, float4 alpha, float4 momenta, float lambda)
{
	int2 hiddenPosition = (int2)(get_global_id(0), get_global_id(1));

	float2 positionNormalized = (float2)(hiddenPosition.x * layerSizeMinusOneInv.x, hiddenPosition.y * layerSizeMinusOneInv.y);
	int2 nextCenterPosition = (int2)(positionNormalized.x * nextSizeMinusOne.x, positionNormalized.y * nextSizeMinusOne.y);
	int2 inputCenterPosition = (int2)(positionNormalized.x * inputSizeMinusOne.x, positionNormalized.y * inputSizeMinusOne.y);

	float2 thisState = read_imagef(hiddenStatesTemporal, hiddenPosition).xy;

	// ------------------------------------ Activate from Reconstruction ------------------------------------

	float sum = 0.0f;

	int wi = 0;

	for (int dx = -predictiveRadius; dx <= predictiveRadius; dx++)
		for (int dy = -predictiveRadius; dy <= predictiveRadius; dy++) {
			int2 layerPosition = (int2)(inputCenterPosition.x + dx, inputCenterPosition.y + dy);

			if (layerPosition.x >= 0 && layerPosition.x < inputSize.x && layerPosition.y >= 0 && layerPosition.y < inputSize.y) {
				float state = read_imagef(hiddenStatesSpatial, layerPosition).x;
				float recon = read_imagef(spatialReconstruction, layerPosition).x;

				float weight = read_imagef(predictiveWeightsPrev, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0)).x;

				sum += weight * (state - recon);
			}

			wi++;
		}

	// Bias
	//float bias = read_imagef(predictiveWeightsPrev, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0)).x;

	//sum += bias;

	wi = 0;

	for (int dx = -feedBackRadius; dx <= feedBackRadius; dx++)
		for (int dy = -feedBackRadius; dy <= feedBackRadius; dy++) {
			int2 nextPosition = (int2)(nextCenterPosition.x + dx, nextCenterPosition.y + dy);

			if (nextPosition.x >= 0 && nextPosition.x < nextSize.x && nextPosition.y >= 0 && nextPosition.y < nextSize.y) {
				float nextState = read_imagef(hiddenStatesNextTemporal, nextPosition).x;
				float nextRecon = read_imagef(nextTemporalReconstruction, nextPosition).x;

				float weight = read_imagef(feedBackWeightsPrev, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0)).x;

				sum += weight * (nextState - nextRecon);
			}

			wi++;
		}

	wi = 0;

	for (int dx = -lateralConnectionRadius; dx <= lateralConnectionRadius; dx++)
		for (int dy = -lateralConnectionRadius; dy <= lateralConnectionRadius; dy++) {
			int2 layerPosition = (int2)(hiddenPosition.x + dx, hiddenPosition.y + dy);

			if (layerPosition.x >= 0 && layerPosition.x < layerSize.x && layerPosition.y >= 0 && layerPosition.y < layerSize.y) {
				float state = read_imagef(hiddenStatesTemporalPrev, layerPosition).x;
				float recon = read_imagef(temporalReconstruction, layerPosition).x;

				float weight = read_imagef(lateralWeightsPrev, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0)).x;

				sum += weight * (state - recon);
			}

			wi++;
		}

	sum *= thisState.y;

	// ------------------------------------ Update ------------------------------------

	wi = 0;

	for (int dx = -predictiveRadius; dx <= predictiveRadius; dx++)
		for (int dy = -predictiveRadius; dy <= predictiveRadius; dy++) {
			int2 layerPosition = (int2)(inputCenterPosition.x + dx, inputCenterPosition.y + dy);

			if (layerPosition.x >= 0 && layerPosition.x < inputSize.x && layerPosition.y >= 0 && layerPosition.y < inputSize.y) {
				float state = read_imagef(hiddenStatesSpatial, layerPosition).x;
				float recon = read_imagef(spatialReconstruction, layerPosition).x;

				float2 prevWeight = read_imagef(predictiveWeightsPrev, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0)).xy;

				float newWeight = prevWeight.x + alpha.x * (0.5f * (sum * state + (state - recon) * thisState.x)) + momenta.x * prevWeight.y;
				float newDelta = newWeight - prevWeight.x;

				write_imagef(predictiveWeights, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0), (float4)(newWeight, newDelta, 0.0f, 0.0f));
			}

			wi++;
		}

	// Bias
	float2 prevWeight = read_imagef(predictiveWeightsPrev, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0)).xy;

	float newWeight = prevWeight.x + alpha.x * (sum) + momenta.x * prevWeight.y;
	float newDelta = newWeight - prevWeight.x;

	write_imagef(predictiveWeights, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0), (float4)(newWeight, newDelta, 0.0f, 0.0f));

	wi = 0;

	for (int dx = -feedBackRadius; dx <= feedBackRadius; dx++)
		for (int dy = -feedBackRadius; dy <= feedBackRadius; dy++) {
			int2 nextPosition = (int2)(nextCenterPosition.x + dx, nextCenterPosition.y + dy);

			if (nextPosition.x >= 0 && nextPosition.x < nextSize.x && nextPosition.y >= 0 && nextPosition.y < nextSize.y) {
				float state = read_imagef(hiddenStatesNextTemporal, nextPosition).x;
				float recon = read_imagef(nextTemporalReconstruction, nextPosition).x;

				float2 prevWeight = read_imagef(feedBackWeightsPrev, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0)).xy;

				float newWeight = prevWeight.x + alpha.y * (0.5f * (sum * state + (state - recon) * thisState.x)) + momenta.y * prevWeight.y;
				float newDelta = newWeight - prevWeight.x;

				write_imagef(feedBackWeights, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0), (float4)(newWeight, newDelta, 0.0f, 0.0f));
			}

			wi++;
		}

	wi = 0;

	for (int dx = -lateralConnectionRadius; dx <= lateralConnectionRadius; dx++)
		for (int dy = -lateralConnectionRadius; dy <= lateralConnectionRadius; dy++) {
			int2 layerPosition = (int2)(hiddenPosition.x + dx, hiddenPosition.y + dy);

			if (layerPosition.x >= 0 && layerPosition.x < layerSize.x && layerPosition.y >= 0 && layerPosition.y < layerSize.y) {
				float state = read_imagef(hiddenStatesTemporalPrev, layerPosition).x;
				float recon = read_imagef(temporalReconstruction, layerPosition).x;

				float2 prevWeight = read_imagef(lateralWeightsPrev, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0)).xy;

				float newWeight = prevWeight.x + alpha.z * (0.5f * (sum * state + (state - recon) * thisState.x)) + momenta.z * prevWeight.y;
				float newDelta = newWeight - prevWeight.x;

				write_imagef(lateralWeights, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0), (float4)(newWeight, newDelta, 0.0f, 0.0f));

			}

			wi++;
		}
}

void kernel layerUpdateTemporalWeightsLast(read_only image2d_t hiddenStatesSpatial, read_only image2d_t hiddenStatesTemporal, read_only image2d_t hiddenStatesTemporalPrev,
	read_only image2d_t spatialReconstruction, read_only image2d_t temporalReconstruction,
	read_only image3d_t predictiveWeightsPrev, read_only image3d_t lateralWeightsPrev,
	write_only image3d_t predictiveWeights, write_only image3d_t lateralWeights,
	int2 layerSize, int2 layerSizeMinusOne, float2 layerSizeMinusOneInv, int2 inputSize, int2 inputSizeMinusOne, float2 inputSizeMinusOneInv,
	int predictiveRadius, int lateralConnectionRadius,
	float sparsity, int inhibitionRadius, float4 alpha, float4 momenta, float lambda)
{
	int2 hiddenPosition = (int2)(get_global_id(0), get_global_id(1));

	float2 positionNormalized = (float2)(hiddenPosition.x * layerSizeMinusOneInv.x, hiddenPosition.y * layerSizeMinusOneInv.y);
	int2 inputCenterPosition = (int2)(positionNormalized.x * inputSizeMinusOne.x, positionNormalized.y * inputSizeMinusOne.y);

	float2 thisState = read_imagef(hiddenStatesTemporal, hiddenPosition).xy;

	// ------------------------------------ Activate from Reconstruction ------------------------------------

	float sum = 0.0f;

	int wi = 0;

	for (int dx = -predictiveRadius; dx <= predictiveRadius; dx++)
		for (int dy = -predictiveRadius; dy <= predictiveRadius; dy++) {
			int2 layerPosition = (int2)(inputCenterPosition.x + dx, inputCenterPosition.y + dy);

			if (layerPosition.x >= 0 && layerPosition.x < inputSize.x && layerPosition.y >= 0 && layerPosition.y < inputSize.y) {
				float state = read_imagef(hiddenStatesSpatial, layerPosition).x;
				float recon = read_imagef(spatialReconstruction, layerPosition).x;

				float weight = read_imagef(predictiveWeightsPrev, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0)).x;

				sum += weight * (state - recon);
			}

			wi++;
		}

	// Bias
	//float bias = read_imagef(predictiveWeightsPrev, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0)).x;

	//sum += bias;

	wi = 0;

	for (int dx = -lateralConnectionRadius; dx <= lateralConnectionRadius; dx++)
		for (int dy = -lateralConnectionRadius; dy <= lateralConnectionRadius; dy++) {
			int2 layerPosition = (int2)(hiddenPosition.x + dx, hiddenPosition.y + dy);

			if (layerPosition.x >= 0 && layerPosition.x < layerSize.x && layerPosition.y >= 0 && layerPosition.y < layerSize.y) {
				float state = read_imagef(hiddenStatesTemporalPrev, layerPosition).x;
				float recon = read_imagef(temporalReconstruction, layerPosition).x;

				float weight = read_imagef(lateralWeightsPrev, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0)).x;

				sum += weight * (state - recon);
			}

			wi++;
		}

	sum *= thisState.y;

	// ------------------------------------ Update ------------------------------------

	wi = 0;

	for (int dx = -predictiveRadius; dx <= predictiveRadius; dx++)
		for (int dy = -predictiveRadius; dy <= predictiveRadius; dy++) {
			int2 layerPosition = (int2)(inputCenterPosition.x + dx, inputCenterPosition.y + dy);

			if (layerPosition.x >= 0 && layerPosition.x < inputSize.x && layerPosition.y >= 0 && layerPosition.y < inputSize.y) {
				float state = read_imagef(hiddenStatesSpatial, layerPosition).x;
				float recon = read_imagef(spatialReconstruction, layerPosition).x;

				float2 prevWeight = read_imagef(predictiveWeightsPrev, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0)).xy;

				float newWeight = prevWeight.x + alpha.x * (0.5f * (sum * state + (state - recon) * thisState.x)) + momenta.x * prevWeight.y;
				float newDelta = newWeight - prevWeight.x;

				write_imagef(predictiveWeights, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0), (float4)(newWeight, newDelta, 0.0f, 0.0f));
			}

			wi++;
		}

	// Bias
	float2 prevWeight = read_imagef(predictiveWeightsPrev, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0)).xy;

	float newWeight = prevWeight.x + alpha.x * sum + momenta.x * prevWeight.y;
	float newDelta = newWeight - prevWeight.x;

	write_imagef(predictiveWeights, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0), (float4)(newWeight, newDelta, 0.0f, 0.0f));

	wi = 0;

	for (int dx = -lateralConnectionRadius; dx <= lateralConnectionRadius; dx++)
		for (int dy = -lateralConnectionRadius; dy <= lateralConnectionRadius; dy++) {
			int2 layerPosition = (int2)(hiddenPosition.x + dx, hiddenPosition.y + dy);

			if (layerPosition.x >= 0 && layerPosition.x < layerSize.x && layerPosition.y >= 0 && layerPosition.y < layerSize.y) {
				float state = read_imagef(hiddenStatesTemporalPrev, layerPosition).x;
				float recon = read_imagef(temporalReconstruction, layerPosition).x;

				float2 prevWeight = read_imagef(lateralWeightsPrev, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0)).xy;

				float newWeight = prevWeight.x + alpha.z * (0.5f * (sum * state + (state - recon) * thisState.x)) + momenta.z * prevWeight.y;
				float newDelta = newWeight - prevWeight.x;

				write_imagef(lateralWeights, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0), (float4)(newWeight, newDelta, 0.0f, 0.0f));
			}

			wi++;
		}
}

void kernel layerSpatialPredictiveReconstructionWeightUpdate(read_only image2d_t visibleReconstruction, read_only image2d_t inputs, read_only image2d_t hiddenStatesPrev, read_only image3d_t reconstructionWeightsPrev, write_only image3d_t reconstructionWeights,
	int reconstructionReceptiveRadius, int2 inputSizeMinusOne, float2 inputSizeMinusOneInv, int2 layerSize, int2 layerSizeMinusOne, float2 layerSizeMinusOneInv, float alpha, float momentum)
{
	int2 visiblePosition = (int2)(get_global_id(0), get_global_id(1));
	float2 layerPositionNormalized = (float2)(visiblePosition.x * inputSizeMinusOneInv.x, visiblePosition.y * inputSizeMinusOneInv.y);
	int2 layerPositionCenter = (int2)(layerPositionNormalized.x * layerSizeMinusOne.x, layerPositionNormalized.y * layerSizeMinusOne.y);

	float input = read_imagef(inputs, visiblePosition).x;
	float recon = read_imagef(visibleReconstruction, visiblePosition).x;

	float error = input - recon;

	int wi = 0;

	for (int dx = -reconstructionReceptiveRadius; dx <= reconstructionReceptiveRadius; dx++)
		for (int dy = -reconstructionReceptiveRadius; dy <= reconstructionReceptiveRadius; dy++) {
			int2 layerPosition = (int2)(layerPositionCenter.x + dx, layerPositionCenter.y + dy);

			if (layerPosition.x >= 0 && layerPosition.x < layerSize.x && layerPosition.y >= 0 && layerPosition.y < layerSize.y) {
				float source = read_imagef(hiddenStatesPrev, layerPosition).x;

				float eligibility = error * source;

				float2 prevWeight = read_imagef(reconstructionWeightsPrev, (int4)(visiblePosition.x, visiblePosition.y, wi, 0)).xy;

				float newWeight = prevWeight.x + alpha * eligibility + momentum * prevWeight.y;
				float newDelta = newWeight - prevWeight.x;

				write_imagef(reconstructionWeights, (int4)(visiblePosition.x, visiblePosition.y, wi, 0), (float4)(newWeight, newDelta, 0.0f, 0.0f));
			}

			wi++;
		}
}

void kernel gaussianBlurX(read_only image2d_t source, write_only image2d_t destination, float2 sizeInv, float kernelWidth) {
	int2 destinationPosition = (int2)(get_global_id(0), get_global_id(1));
	float2 destinationPositionNormalized = (float2)(destinationPosition.x * sizeInv.x, destinationPosition.y * sizeInv.y);

	float4 sum = (float4)(0.0f, 0.0f, 0.0f, 0.0f);

	sum += read_imagef(source, defaultNormalizedSampler, (float2)(destinationPositionNormalized.x - 4.0f * kernelWidth, destinationPositionNormalized.y)) * 0.05f;
	sum += read_imagef(source, defaultNormalizedSampler, (float2)(destinationPositionNormalized.x - 3.0f * kernelWidth, destinationPositionNormalized.y)) * 0.09f;
	sum += read_imagef(source, defaultNormalizedSampler, (float2)(destinationPositionNormalized.x - 2.0f * kernelWidth, destinationPositionNormalized.y)) * 0.12f;
	sum += read_imagef(source, defaultNormalizedSampler, (float2)(destinationPositionNormalized.x - kernelWidth, destinationPositionNormalized.y)) * 0.15f;
	sum += read_imagef(source, defaultNormalizedSampler, (float2)(destinationPositionNormalized.x, destinationPositionNormalized.y)) * 0.16f;
	sum += read_imagef(source, defaultNormalizedSampler, (float2)(destinationPositionNormalized.x + kernelWidth, destinationPositionNormalized.y)) * 0.15f;
	sum += read_imagef(source, defaultNormalizedSampler, (float2)(destinationPositionNormalized.x + 2.0f * kernelWidth, destinationPositionNormalized.y)) * 0.12f;
	sum += read_imagef(source, defaultNormalizedSampler, (float2)(destinationPositionNormalized.x + 3.0f * kernelWidth, destinationPositionNormalized.y)) * 0.09f;
	sum += read_imagef(source, defaultNormalizedSampler, (float2)(destinationPositionNormalized.x + 4.0f * kernelWidth, destinationPositionNormalized.y)) * 0.05f;

	write_imagef(destination, destinationPosition, sum);
}

void kernel gaussianBlurY(read_only image2d_t source, write_only image2d_t destination, float2 sizeInv, float kernelWidth) {
	int2 destinationPosition = (int2)(get_global_id(0), get_global_id(1));
	float2 destinationPositionNormalized = (float2)(destinationPosition.x * sizeInv.x, destinationPosition.y * sizeInv.y);

	float4 sum = (float4)(0.0f, 0.0f, 0.0f, 0.0f);

	sum += read_imagef(source, defaultNormalizedSampler, (float2)(destinationPositionNormalized.x, destinationPositionNormalized.y - 4.0f * kernelWidth)) * 0.05f;
	sum += read_imagef(source, defaultNormalizedSampler, (float2)(destinationPositionNormalized.x, destinationPositionNormalized.y - 3.0f * kernelWidth)) * 0.09f;
	sum += read_imagef(source, defaultNormalizedSampler, (float2)(destinationPositionNormalized.x, destinationPositionNormalized.y - 2.0f * kernelWidth)) * 0.12f;
	sum += read_imagef(source, defaultNormalizedSampler, (float2)(destinationPositionNormalized.x, destinationPositionNormalized.y - kernelWidth)) * 0.15f;
	sum += read_imagef(source, defaultNormalizedSampler, (float2)(destinationPositionNormalized.x, destinationPositionNormalized.y)) * 0.16f;
	sum += read_imagef(source, defaultNormalizedSampler, (float2)(destinationPositionNormalized.x, destinationPositionNormalized.y + kernelWidth)) * 0.15f;
	sum += read_imagef(source, defaultNormalizedSampler, (float2)(destinationPositionNormalized.x, destinationPositionNormalized.y + 2.0f * kernelWidth)) * 0.12f;
	sum += read_imagef(source, defaultNormalizedSampler, (float2)(destinationPositionNormalized.x, destinationPositionNormalized.y + 3.0f * kernelWidth)) * 0.09f;
	sum += read_imagef(source, defaultNormalizedSampler, (float2)(destinationPositionNormalized.x, destinationPositionNormalized.y + 4.0f * kernelWidth)) * 0.05f;

	write_imagef(destination, destinationPosition, sum);
}