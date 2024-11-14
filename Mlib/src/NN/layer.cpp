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
		//setup biases and gradients with random values or 0s
		for (int bias = 0; bias < numAft; bias++)
		{
			if (rand)
				biases.push_back(static_cast<float>(random(0, 100'000) / 100'000.f - 0.5));
			else
				biases.push_back(0);

			//gradient is always 0
			biasesGradients.push_back(0);
		}
		biasesVelocities = biasesGradients;
		//setup weights and gradients with random values or 0s
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

				//gradient is always 0
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

		//load all weights and gradients
		for (int bef = 0; bef < numBef; bef++)
		{
			std::vector<float> partialWeights;
			std::vector<float> partialWeightsGradients;
			for (int aft = 0; aft < numAft; aft++)
			{
				getline(layerStrean, token, ',');
				partialWeights.push_back(static_cast<float>(stod(token)));

				//gradient is always 0
				partialWeightsGradients.push_back(0);
			}
			weights.push_back(partialWeights);
			weightsGradients.push_back(partialWeightsGradients);
		}
		weightsVelocities = weightsGradients;

		//load all biases and gradients
		for (int bias = 0; bias < numAft; bias++)
		{
			getline(layerStrean, token, ',');
			biases.push_back(static_cast<float>(stod(token)));

			//gradient is always 0s
			biasesGradients.push_back(0);
		}
		biasesVelocities = biasesGradients;
	}

	std::vector<float> NN::Layer::forwardPass(const std::vector<float>& inputs, bool output)
	{
		//sizes do not match
		if (inputs.size() != numBef) {
			std::cout << "ERROR: layer input size: " << numBef << ", data size: " << inputs.size();
			std::exit(-103);
		}

		//store value for later
		inputValues = inputs;

		weightedValues.clear();
		//calculate weighted values (nodesBefore * weights + bias)
		for (int aft = 0; aft < numAft; aft++)
		{
			float weightedValue = biases[aft];
			for (int bef = 0; bef < numBef; bef++)
				weightedValue += inputs[bef] * weights[bef][aft];
			weightedValues.push_back(weightedValue);
		}

		//pass the weighted values through the chosen activation function
		if (!output)
			activatedValues = hiddenAct(weightedValues);
		else
			activatedValues = outputAct(weightedValues);

		return activatedValues;
	}
	std::vector<float> NN::Layer::computeHiddenNodeValues(const std::vector<float>& nodeValuesAft, const Layer& layerAft) const
	{
		std::vector<float> nodeValues;
		std::vector<float> actDer = hiddenActDer(weightedValues);

		//no idead wtf is going on here
		for (int aft = 0; aft < numAft; aft++)
		{
			float nodeValue = 0;
			for (int aftAft = 0; aftAft < layerAft.numAft; aftAft++)
			{
				float inputDerivative = layerAft.weights[aft][aftAft];
				nodeValue += inputDerivative * nodeValuesAft[aftAft];
			}
			nodeValues.push_back(nodeValue * actDer[aft]);
		}

		return nodeValues;
	}
	std::vector<float> NN::Layer::computeOutputNodeValues(const std::vector<float>& targets) const
	{
		std::vector<float> nodeValues;

		//special case, return immediately after
		if (outAct == ActFunc::Softmax && lossFunc == LossFunc::CrossEntropy)
		{
			for (int i = 0; i < activatedValues.size(); i++)
				nodeValues.push_back(activatedValues[i] - targets[i]);

			return nodeValues;
		}

		//calculate output activation derivatives according to the chosen function
		if (outAct == ActFunc::Sigmoid)
		{
			for (const auto& v : activatedValues)
			{
				float sigm = 1.f / (1 + exp(-v));
				nodeValues.push_back(sigm * (1 - sigm));
			}
		}
		else 
			std::exit(-101);

		//calculate loss derivative according to the chosen function
		if (lossFunc == LossFunc::SquaredError)
		{
			for (int i = 0; i < activatedValues.size(); i++)
				nodeValues[i] *= 2 * (activatedValues[i] - targets[i]);
		}
		else 
			std::exit(-101);

		return nodeValues;
	}

	void NN::Layer::updateGradients(const std::vector<float>& nodeValues)
	{
		//update weights gradients
		for (int bef = 0; bef < numBef; bef++)
		{
			for (int aft = 0; aft < numAft; aft++)
			{
				float weightDerivative = inputValues[bef] * nodeValues[aft];
				weightsGradients[bef][aft] += weightDerivative;
			}
		}

		//update biases gradients
		for (int aft = 0; aft < numAft; aft++)
			biasesGradients[aft] += nodeValues[aft];
	}
	void NN::Layer::applyGradients(float learnRate, float momentum, int batchSize)
	{
		//apply all the weights gradients
		for (int bef = 0; bef < numBef; bef++)
		{
			for (int aft = 0; aft < numAft; aft++)
			{
				weightsGradients[bef][aft] /= float(batchSize);
				//find new velocity based on the momentum coefficient
				weightsVelocities[bef][aft] = momentum * weightsVelocities[bef][aft] + 
					(1 - momentum) * weightsGradients[bef][aft];
				//apply the velocity based on the learn rate
				weights[bef][aft] -= weightsVelocities[bef][aft] * learnRate;
			}
		}
		
		//apply all the biases gradients
		for (int aft = 0; aft < numAft; aft++)
		{
			biasesGradients[aft] /= float(batchSize);
			//find new velocity based on the momentum coefficient
			biasesVelocities[aft] = momentum * biasesVelocities[aft] + (1 - momentum) * biasesGradients[aft];
			//apply the velocity based on the learn rate
			biases[aft] -= biasesVelocities[aft] * learnRate;
		}
	}
	void NN::Layer::clearGradients() 
	{
		//clear all weights gradients
		for (int bef = 0; bef < numBef; bef++)
		{
			for (int aft = 0; aft < numAft; aft++)
				weightsGradients[bef][aft] = 0;
		}

		//clear all biases gradients
		for (int aft = 0; aft < numAft; aft++)
			biasesGradients[aft] = 0;
	}

	float NN::Layer::loss(const std::vector<float>& values, const std::vector<float>& targets) const
	{
		//sizes do not match
		if (values.size() != targets.size()) {
			std::cout << "ERROR: layer output size: " << values.size() << ", targets size: " << targets.size();
			std::exit(-103);
		}

		float loss = 0.0;

		//calculate loss according to the chosen function
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
		else 
			std::exit(-102);

		return loss;
	}

	std::string NN::Layer::toString() const
	{
		std::ostringstream ss;

		//add weights to the stream
		for (int bef = 0; bef < numBef; bef++)
		{
			for (int aft = 0; aft < numAft; aft++)
				ss << std::to_string(weights[bef][aft]) << ',';
		}

		//add biases to the stream
		for (int aft = 0; aft < numAft; aft++)
			ss << std::to_string(biases[aft]) << ',';

		return ss.str();
	}

	std::vector<float> NN::Layer::hiddenAct(const std::vector<float>& values) const
	{
		std::vector<float> activated;

		//calculate hidden activated values according to the chosen function
		if (hidAct == ActFunc::Sigmoid)
		{
			for (const auto& v : values)
				activated.push_back(1.f / (1 + exp(-v)));
		}
		else if (hidAct == ActFunc::ReLU)
		{
			for (const auto& v : values)
				activated.push_back(std::max(v, float(0)));
		}
		else
			std::exit(-100);

		return activated;
	}
	std::vector<float> NN::Layer::hiddenActDer(const std::vector<float>& values) const
	{
		std::vector<float> derivatives;

		//calculate hidden activation derivatives according to the chosen function
		if (hidAct == ActFunc::Sigmoid)
		{
			for (const auto& v : values)
			{
				float activated = 1.f / (1 + exp(-v));
				derivatives.push_back(activated * (1 - activated));
			}
		}
		else if (hidAct == ActFunc::ReLU)
		{
			for (const auto& v : values)
			{
				if (v < 0)
					derivatives.push_back(0);
				else
					derivatives.push_back(1);
			}
		}
		else
			std::exit(-101);

		return derivatives;
	}
	std::vector<float> NN::Layer::outputAct(const std::vector<float>& values) const
	{
		std::vector<float> activated;

		//calculate output activated values according to the chosen function
		if (outAct == ActFunc::Sigmoid)
		{
			for (const auto& v : values)
				activated.push_back(1.f / (1 + exp(-v)));
		}
		else if (outAct == ActFunc::Softmax)
		{
			float expSum = 0;
			for (const auto& v : values) {
				float exp = std::exp(v);
				activated.push_back(exp);
				expSum += exp;
			}
			for (float& v : activated)
				v /= expSum;
		}
		else
			std::exit(-100);

		return activated;
	}
}