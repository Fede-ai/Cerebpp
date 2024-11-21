#include "Crb/FNN/layer.hpp"
#include <random>
#include <sstream>

namespace Crb {
	FNN::Layer::Layer(int inNumBef, int inNumAft, bool rand, ActFunc inHidAct, ActFunc inOutAct, LossFunc inLossFunc)
		:
		numBefore(inNumBef),
		numAfter(inNumAft),
		hidAct(inHidAct),
		outAct(inOutAct),
		lossFunc(inLossFunc)
	{
		static std::random_device dev;
		static std::mt19937 rng(dev());
		std::uniform_int_distribution<std::mt19937::result_type> dist(0, 100'000);

		//setup biases and gradients with random values or 0s
		for (int bias = 0; bias < numAfter; bias++) {
			if (rand)
				biases.push_back(dist(rng) / 100'000.f - 0.5f);
			else
				biases.push_back(0);

			//gradient is always 0
			biasesGradients.push_back(0);
		}
		biasesVelocities = biasesGradients;
		//setup weights and gradients with random values or 0s
		for (int bef = 0; bef < numBefore; bef++) {
			std::vector<float> partialWeights;
			std::vector<float> partialWeightsGradients;
			for (int aft = 0; aft < numAfter; aft++) {
				if (rand)
					partialWeights.push_back(dist(rng) / 100'000.f - 0.5f);
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
	FNN::Layer::Layer(int inNumBef, int inNumAft, ActFunc inHidAct, ActFunc inOutAct, LossFunc inLossFunc, std::string string)
		:
		numBefore(inNumBef),
		numAfter(inNumAft),
		hidAct(inHidAct),
		outAct(inOutAct),
		lossFunc(inLossFunc)
	{
		std::string token;
		std::istringstream layerStrean(string);

		//load all weights and gradients
		for (int bef = 0; bef < numBefore; bef++) {
			std::vector<float> partialWeights;
			std::vector<float> partialWeightsGradients;
			for (int aft = 0; aft < numAfter; aft++) {
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
		for (int bias = 0; bias < numAfter; bias++) {
			getline(layerStrean, token, ',');
			biases.push_back(static_cast<float>(stod(token)));

			//gradient is always 0s
			biasesGradients.push_back(0);
		}
		biasesVelocities = biasesGradients;
	}

	std::vector<float> FNN::Layer::forwardPass(const std::vector<float>& inputs, bool output)
	{
		//sizes do not match
		if (inputs.size() != numBefore)
			throw std::runtime_error("data size != layer input size");

		//store value for later
		inputValues = inputs;

		weightedValues.clear();
		//calculate weighted values (nodesBefore * weights + bias)
		for (int aft = 0; aft < numAfter; aft++) {
			float weightedValue = biases[aft];
			for (int bef = 0; bef < numBefore; bef++)
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
	std::vector<float> FNN::Layer::computeHiddenNodeValues(const std::vector<float>& nodeValuesAft, const Layer& layerAft) const
	{
		std::vector<float> nodeValues;

		//calculate hidden activation derivatives according to the chosen function
		if (hidAct == ActFunc::Sigmoid)
		{
			for (const auto& v : weightedValues) {
				float activated = 1.f / (1 + exp(-v));
				nodeValues.push_back(activated * (1 - activated));
			}
		}
		else if (hidAct == ActFunc::ReLU)
		{
			for (const auto& v : weightedValues) {
				if (v < 0)
					nodeValues.push_back(0);
				else
					nodeValues.push_back(1);
			}
		}
		else
			throw std::runtime_error("no valid hidden activation function");

		//no idead wtf is going on here
		for (int aft = 0; aft < numAfter; aft++) {
			float nodeValue = 0;
			for (int aftAft = 0; aftAft < layerAft.numAfter; aftAft++)
				nodeValue += layerAft.weights[aft][aftAft] * nodeValuesAft[aftAft];
			nodeValues[aft] *= nodeValue;
		}

		return nodeValues;
	}
	std::vector<float> FNN::Layer::computeOutputNodeValues(const std::vector<float>& targets) const
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
			throw std::runtime_error("no valid output activation function");

		//calculate loss derivative according to the chosen function
		if (lossFunc == LossFunc::SquaredError)
		{
			for (int i = 0; i < activatedValues.size(); i++)
				nodeValues[i] *= 2 * (activatedValues[i] - targets[i]);
		}
		else 
			throw std::runtime_error("no valid loss function");

		return nodeValues;
	}

	void FNN::Layer::updateGradients(const std::vector<float>& nodeValues)
	{
		//update weights gradients
		for (int bef = 0; bef < numBefore; bef++) {
			for (int aft = 0; aft < numAfter; aft++)
				weightsGradients[bef][aft] += inputValues[bef] * nodeValues[aft];
		}

		//update biases gradients
		for (int aft = 0; aft < numAfter; aft++)
			biasesGradients[aft] += nodeValues[aft];
	}
	void FNN::Layer::applyGradients(float learnRate, float momentum, int batchSize)
	{
		//apply all the weights gradients
		for (int bef = 0; bef < numBefore; bef++) {
			for (int aft = 0; aft < numAfter; aft++) {
				weightsGradients[bef][aft] /= float(batchSize);
				//find new velocity based on the momentum coefficient
				weightsVelocities[bef][aft] = momentum * weightsVelocities[bef][aft] + 
					(1 - momentum) * weightsGradients[bef][aft];
				//apply the velocity based on the learn rate
				weights[bef][aft] -= weightsVelocities[bef][aft] * learnRate;
			}
		}
		
		//apply all the biases gradients
		for (int aft = 0; aft < numAfter; aft++) {
			biasesGradients[aft] /= float(batchSize);
			//find new velocity based on the momentum coefficient
			biasesVelocities[aft] = momentum * biasesVelocities[aft] + (1 - momentum) * biasesGradients[aft];
			//apply the velocity based on the learn rate
			biases[aft] -= biasesVelocities[aft] * learnRate;
		}
	}
	void FNN::Layer::clearGradients() 
	{
		//clear all weights gradients
		for (int bef = 0; bef < numBefore; bef++) {
			for (int aft = 0; aft < numAfter; aft++)
				weightsGradients[bef][aft] = 0;
		}

		//clear all biases gradients
		for (int aft = 0; aft < numAfter; aft++)
			biasesGradients[aft] = 0;
	}

	float FNN::Layer::loss(const std::vector<float>& values, const std::vector<float>& targets) const
	{
		//sizes do not match
		if (values.size() != targets.size())
			throw std::runtime_error("targets size != layer output size");

		float loss = 0.0;

		//calculate loss according to the chosen function
		if (lossFunc == LossFunc::SquaredError) {
			for (int i = 0; i < values.size(); i++)
				loss += static_cast<float>(std::pow((targets[i] - values[i]), 2));
		}
		else if (lossFunc == LossFunc::CrossEntropy) {
			for (int i = 0; i < values.size(); i++)
				loss += -targets[i] * static_cast<float>(std::log(values[i] + 1e-15));
		}
		else 
			throw std::runtime_error("no valid loss function");

		return loss;
	}

	std::string FNN::Layer::toString() const
	{
		std::ostringstream ss;

		//add weights to the stream
		for (int bef = 0; bef < numBefore; bef++) {
			for (int aft = 0; aft < numAfter; aft++)
				ss << std::to_string(weights[bef][aft]) << ',';
		}

		//add biases to the stream
		for (int aft = 0; aft < numAfter; aft++)
			ss << std::to_string(biases[aft]) << ',';

		return ss.str();
	}

	std::vector<float> FNN::Layer::hiddenAct(const std::vector<float>& values) const
	{
		std::vector<float> activated;

		//calculate hidden activated values according to the chosen function
		if (hidAct == ActFunc::Sigmoid) {
			for (const auto& v : values)
				activated.push_back(1.f / (1 + exp(-v)));
		}
		else if (hidAct == ActFunc::ReLU) {
			for (const auto& v : values)
				activated.push_back(std::max(v, float(0)));
		}
		else
			throw std::runtime_error("no valid hidden activation function");

		return activated;
	}
	std::vector<float> FNN::Layer::outputAct(const std::vector<float>& values) const
	{
		std::vector<float> activated;

		//calculate output activated values according to the chosen function
		if (outAct == ActFunc::Sigmoid) {
			for (const auto& v : values)
				activated.push_back(1.f / (1 + exp(-v)));
		}
		else if (outAct == ActFunc::Softmax) {
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
			throw std::runtime_error("no valid output activation function");

		return activated;
	}
}