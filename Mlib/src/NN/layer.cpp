#include "nn.hpp"
#include "../Utility/math.hpp"
#include <iostream>

namespace Mlib {
	NN::Layer::Layer(int inNumBef, int inNumAft, bool rand, ActFunc inHidAct, ActFunc inOutAct, LossFunc inLossFunc)
		:
		numBef(inNumBef),
		numAft(inNumAft),
		hidAct(inHidAct),
		outAct(inOutAct),
		lossFunc(inLossFunc)
	{
		//setup biases
		for (int bias = 0; bias < numAft; bias++)
		{
			if (rand)
				biases.push_back(static_cast<float>(random(0, 100'000) / 100'000.f - 0.5));
			else
				biases.push_back(0);
			biasesGradients.push_back(0);
		}
		biasesVelocities = biasesGradients;
		//setup weights
		for (int bef = 0; bef < numBef; bef++)
		{
			std::vector<float> partialWeights;
			std::vector<float> partialWeightsGradients;
			for (int aft = 0; aft < numAft; aft++)
			{
				if (rand)
					partialWeights.push_back(static_cast<float>(random(0, 100'000) / 100'000.f - 0.5));
				else
					partialWeights.push_back(0);
				partialWeightsGradients.push_back(0);
			}
			weights.push_back(partialWeights);
			weightsGradients.push_back(partialWeightsGradients);
		}
		weightsVelocities = weightsGradients;
	}
	NN::Layer::Layer(int inNumBef, int inNumAft, ActFunc inHidAct, ActFunc inOutAct, LossFunc inLossFunc, std::string string)
		:
		numBef(inNumBef),
		numAft(inNumAft),
		hidAct(inHidAct),
		outAct(inOutAct),
		lossFunc(inLossFunc)
	{
		std::string token;
		std::istringstream layerStrean(string);

		//setup weights
		for (int bef = 0; bef < numBef; bef++)
		{
			std::vector<float> partialWeights;
			std::vector<float> partialWeightsGradients;
			for (int aft = 0; aft < numAft; aft++)
			{
				getline(layerStrean, token, ',');
				partialWeights.push_back(static_cast<float>(stod(token)));
				partialWeightsGradients.push_back(0);
			}
			weights.push_back(partialWeights);
			weightsGradients.push_back(partialWeightsGradients);
		}
		weightsVelocities = weightsGradients;

		//setup biases
		for (int bias = 0; bias < numAft; bias++)
		{
			getline(layerStrean, token, ',');
			biases.push_back(static_cast<float>(stod(token)));
			biasesGradients.push_back(0);
		}
		biasesVelocities = biasesGradients;
	}

	std::vector<float> NN::Layer::computeHidden(std::vector<float> inputs)
	{
		if (inputs.size() != numBef) {
			std::cout << "ERROR: layer input size: " << numBef << ", data size: " << inputs.size();
			std::exit(-103);
		}

		inputValues = inputs;

		weightedValues.clear();
		for (int aft = 0; aft < numAft; aft++)
		{
			float weightedValue = biases[aft];
			for (int bef = 0; bef < numBef; bef++)
				weightedValue += inputs[bef] * weights[bef][aft];
			weightedValues.push_back(weightedValue);
		}

		activatedValues = hiddenAct(weightedValues);
		return activatedValues;
	}
	std::vector<float> NN::Layer::computeHiddenNodeValues(std::vector<float> nodeValuesAfter, Layer layerAft) const
	{
		std::vector<float> nodeValues;
		std::vector<float> actDer = hiddenActDer(weightedValues);

		for (int aft = 0; aft < numAft; aft++)
		{
			float nodeValue = 0;
			for (int aftAft = 0; aftAft < layerAft.numAft; aftAft++)
			{
				float inputDerivative = layerAft.weights[aft][aftAft];
				nodeValue += inputDerivative * nodeValuesAfter[aftAft];
			}
			nodeValues.push_back(nodeValue * actDer[aft]);
		}

		return nodeValues;
	}
	std::vector<float> NN::Layer::computeOutput(std::vector<float> inputs)
	{
		if (inputs.size() != numBef) {
			std::cout << "ERROR: layer input size: " << numBef << ", data size: " << inputs.size();
			std::exit(-103);
		}

		inputValues = inputs;

		weightedValues.clear();
		for (int aft = 0; aft < numAft; aft++)
		{
			float weightedValue = biases[aft];
			for (int bef = 0; bef < numBef; bef++)
				weightedValue += inputs[bef] * weights[bef][aft];
			weightedValues.push_back(weightedValue);
		}

		activatedValues = outputAct(weightedValues);
		return activatedValues;
	}
	std::vector<float> NN::Layer::computeOutputNodeValues(std::vector<float> targets) const
	{
		std::vector<float> nodeValues;
		std::vector<float> lossAndActDer = lossAndOutputActDer(activatedValues, targets);

		for (int aft = 0; aft < numAft; aft++)
			nodeValues.push_back(lossAndActDer[aft]);

		return nodeValues;
	}

	void NN::Layer::updateGradients(std::vector<float> nodeValues)
	{
		for (int bef = 0; bef < numBef; bef++)
		{
			for (int aft = 0; aft < numAft; aft++)
			{
				float weightDerivative = inputValues[bef] * nodeValues[aft];
				weightsGradients[bef][aft] += weightDerivative;
			}
		}

		for (int aft = 0; aft < numAft; aft++)
		{
			biasesGradients[aft] += nodeValues[aft];
		}
	}
	void NN::Layer::applyGradients(float learnRate, float momentum, int batchSize)
	{
		for (int bef = 0; bef < numBef; bef++)
		{
			for (int aft = 0; aft < numAft; aft++)
			{
				weightsGradients[bef][aft] /= float(batchSize);
				weightsVelocities[bef][aft] = momentum * weightsVelocities[bef][aft] + (1 - momentum) * weightsGradients[bef][aft];
				weights[bef][aft] -= weightsVelocities[bef][aft] * learnRate;
			}
		}
		
		for (int aft = 0; aft < numAft; aft++)
		{
			biasesGradients[aft] /= float(batchSize);
			biasesVelocities[aft] = momentum * biasesVelocities[aft] + (1 - momentum) * biasesGradients[aft];
			biases[aft] -= biasesVelocities[aft] * learnRate;
		}
	}
	void NN::Layer::clearGradients() 
	{
		for (int bef = 0; bef < numBef; bef++)
		{
			for (int aft = 0; aft < numAft; aft++)
			{
				weightsGradients[bef][aft] = 0;
			}
		}

		for (int aft = 0; aft < numAft; aft++)
		{
			biasesGradients[aft] = 0;
		}
	}

	float NN::Layer::loss(std::vector<float> values, std::vector<float> targets) const
	{
		if (values.size() != targets.size()) {
			std::cout << "ERROR: layer output size: " << values.size() << ", targets size: " << targets.size();
			std::exit(-103);
		}

		float loss = 0.0;

		if (lossFunc == LossFunc::SquaredError)
		{
			for (int i = 0; i < values.size(); i++)
				loss += static_cast<float>(std::pow((targets[i] - values[i]), 2));
		}
		else if (lossFunc == LossFunc::CrossEntropy)
		{
			for (int i = 0; i < values.size(); i++)
				loss += -targets[i] * static_cast<float>(std::log(values[i] + 1e-15));
		}
		else std::exit(-102);

		return loss;
	}

	std::string NN::Layer::toString() const
	{
		std::ostringstream ss;

		for (int bef = 0; bef < numBef; bef++)
		{
			for (int aft = 0; aft < numAft; aft++)
				ss << std::to_string(weights[bef][aft]) << ',';
		}

		for (int aft = 0; aft < numAft; aft++)
			ss << std::to_string(biases[aft]) << ',';

		return ss.str();
	}

	std::vector<float> NN::Layer::hiddenAct(std::vector<float> values) const
	{
		std::vector<float> activated;

		if (hidAct == ActFunc::Sigmoid)
		{
			for (auto v : values)
				activated.push_back(1.f / (1 + exp(-v)));
		}
		else if (hidAct == ActFunc::ReLU)
		{
			for (auto v : values)
				activated.push_back(std::max(v, float(0)));
		}
		else
		{
			std::exit(-100);
		}

		return activated;
	}
	std::vector<float> NN::Layer::hiddenActDer(std::vector<float> values) const
	{
		std::vector<float> derivatives;

		if (hidAct == ActFunc::Sigmoid)
		{
			for (auto v : values)
			{
				float activated = 1.f / (1 + exp(-v));
				derivatives.push_back(activated * (1 - activated));
			}
		}
		else if (hidAct == ActFunc::ReLU)
		{
			for (auto v : values)
			{
				if (v < 0)
					derivatives.push_back(0);
				else
					derivatives.push_back(1);
			}
		}
		else
		{
			std::exit(-101);
		}

		return derivatives;
	}
	std::vector<float> NN::Layer::outputAct(std::vector<float> values) const
	{
		std::vector<float> activated;

		if (outAct == ActFunc::Sigmoid)
		{
			for (auto v : values)
				activated.push_back(1.f / (1 + exp(-v)));
		}
		else if (outAct == ActFunc::Softmax)
		{
			float expSum = 0;
			for (float v : values) {
				float exp = std::exp(v);
				activated.push_back(exp);
				expSum += exp;
			}
			for (float& v : activated) {
				v /= expSum;
			}
		}
		else
		{
			std::exit(-100);
		}

		return activated;
	}
	std::vector<float> NN::Layer::lossAndOutputActDer(std::vector<float> values, std::vector<float> targets) const
	{
		std::vector<float> nodesLossesDer;

		if (outAct == ActFunc::Softmax && lossFunc == LossFunc::CrossEntropy)
		{
			for (int i = 0; i < values.size(); i++)
				nodesLossesDer.push_back(values[i] - targets[i]);
		}
		else if (outAct == ActFunc::Sigmoid && lossFunc == LossFunc::SquaredError)
		{
			for (int i = 0; i < values.size(); i++)
			{
				float sigm = 1.f / (1 + exp(-values[i]));
				nodesLossesDer.push_back(sigm * (1 - sigm) * 2 * (values[i] - targets[i]));
			}
		}
		else std::exit(-101);

		return nodesLossesDer;
	}
}