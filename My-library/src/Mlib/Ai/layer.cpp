#include "layer.hpp"
#include "../util.hpp"
#include <cmath>

namespace Mlib {
	Layer::Layer(int inNumBef, int inNumAft, bool rand, ActFunc inHidAct, ActFunc inOutAct, LossFunc inLossFunc)
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
				biases.push_back(random(0, 100'000) / 100'000.f - 0.5);
			else
				biases.push_back(0);
			biasesGradients.push_back(0);
		}
		biasesVelocities = biasesGradients;
		//setup weights
		for (int bef = 0; bef < numBef; bef++)
		{
			std::vector<double> partialWeights;
			std::vector<double> partialWeightsGradients;
			for (int aft = 0; aft < numAft; aft++)
			{
				if (rand)
					partialWeights.push_back(random(0, 100'000) / 100'000.f - 0.5);
				else
					partialWeights.push_back(0);
				partialWeightsGradients.push_back(0);
			}
			weights.push_back(partialWeights);
			weightsGradients.push_back(partialWeightsGradients);
		}
		weightsVelocities = weightsGradients;
	}

	std::vector<double> Layer::computeHidden(std::vector<double> inputs)
	{
		if (inputs.size() != numBef)
			std::exit(1000);

		inputValues = inputs;

		weightedValues.clear();
		for (int aft = 0; aft < numAft; aft++)
		{
			double weightedValue = biases[aft];
			for (int bef = 0; bef < numBef; bef++)
				weightedValue += inputs[bef] * weights[bef][aft];
			weightedValues.push_back(weightedValue);
		}

		activatedValues = hiddenAct(weightedValues);
		return activatedValues;
	}
	std::vector<double> Layer::computeHiddenNodeValues(std::vector<double> nodeValuesAfter, Layer layerAft) const
	{
		std::vector<double> nodeValues;
		std::vector<double> actDer = hiddenActDer(weightedValues);

		for (int aft = 0; aft < numAft; aft++)
		{
			double nodeValue = 0;
			for (int aftAft = 0; aftAft < layerAft.numAft; aftAft++)
			{
				double inputDerivative = layerAft.weights[aft][aftAft];
				nodeValue += inputDerivative * nodeValuesAfter[aftAft];
			}
			nodeValues.push_back(nodeValue * actDer[aft]);
		}

		return nodeValues;
	}
	std::vector<double> Layer::computeOutput(std::vector<double> inputs)
	{
		if (inputs.size() != numBef)
			std::exit(1000);

		inputValues = inputs;

		weightedValues.clear();
		for (int aft = 0; aft < numAft; aft++)
		{
			double weightedValue = biases[aft];
			for (int bef = 0; bef < numBef; bef++)
				weightedValue += inputs[bef] * weights[bef][aft];
			weightedValues.push_back(weightedValue);
		}

		activatedValues = outputAct(weightedValues);
		return activatedValues;
	}
	std::vector<double> Layer::computeOutputNodeValues(std::vector<double> targets) const
	{
		std::vector<double> nodeValues;
		std::vector<double> lossAndActDer = lossAndOutputActDer(activatedValues, targets);

		for (int aft = 0; aft < numAft; aft++)
			nodeValues.push_back(lossAndActDer[aft]);

		return nodeValues;
	}

	void Layer::updateGradients(std::vector<double> nodeValues)
	{
		for (int bef = 0; bef < numBef; bef++)
		{
			for (int aft = 0; aft < numAft; aft++)
			{
				double weightDerivative = inputValues[bef] * nodeValues[aft];
				weightsGradients[bef][aft] += weightDerivative;
			}
		}

		for (int aft = 0; aft < numAft; aft++)
		{
			biasesGradients[aft] += nodeValues[aft];
		}
	}
	void Layer::applyGradients(double learnRate, double momentum, int batchSize)
	{
		for (int bef = 0; bef < numBef; bef++)
		{
			for (int aft = 0; aft < numAft; aft++)
			{
				weightsGradients[bef][aft] /= double(batchSize);
				weightsVelocities[bef][aft] = momentum * weightsVelocities[bef][aft] + (1 - momentum) * weightsGradients[bef][aft];
				weights[bef][aft] -= weightsVelocities[bef][aft] * learnRate;
			}
		}
		
		for (int aft = 0; aft < numAft; aft++)
		{
			biasesGradients[aft] /= double(batchSize);
			biasesVelocities[aft] = momentum * biasesVelocities[aft] + (1 - momentum) * biasesGradients[aft];
			biases[aft] -= biasesVelocities[aft] * learnRate;
		}
	}
	void Layer::clearGradients() 
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

	double Layer::loss(std::vector<double> values, std::vector<double> targets) const
	{
		double loss = 0.0;

		if (lossFunc == LossFunc::SquaredError)
		{
			for (int i = 0; i < values.size(); i++)
				loss += std::pow((targets[i] - values[i]), 2);
		}
		else if (lossFunc == LossFunc::CrossEntropy)
		{
			for (int i = 0; i < values.size(); i++)
				loss += -targets[i] * std::log(values[i] + 1e-15);
		}
		else std::exit(-100);

		return loss;
	}

	std::vector<double> Layer::hiddenAct(std::vector<double> values) const
	{
		std::vector<double> activated;

		if (hidAct == ActFunc::Sigmoid)
		{
			for (auto v : values)
				activated.push_back(1.f / (1 + exp(-v)));
		}
		else if (hidAct == ActFunc::ReLU)
		{
			for (auto v : values)
				activated.push_back(std::max(v, double(0)));
		}
		else
		{
			std::exit(-100);
		}

		return activated;
	}
	std::vector<double> Layer::hiddenActDer(std::vector<double> values) const
	{
		std::vector<double> derivatives;

		if (hidAct == ActFunc::Sigmoid)
		{
			for (auto v : values)
			{
				double activated = 1.f / (1 + exp(-v));
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
			std::exit(-100);
		}
			
		return derivatives;
	}
	std::vector<double> Layer::outputAct(std::vector<double> values) const
	{
		std::vector<double> activated;
	
		if (outAct == ActFunc::Sigmoid)
		{
			for (auto v : values)
				activated.push_back(1.f / (1 + exp(-v)));
		}
		else if (outAct == ActFunc::Softmax)
		{
			double expSum = 0;
			for (double v : values) {
				double exp = std::exp(v);	
				activated.push_back(exp);
				expSum += exp;
			}
			for (double& v : activated) {
				v /= expSum;
			}
		}
		else
		{
			std::exit(-100);
		}

		return activated;
	}
	std::vector<double> Layer::lossAndOutputActDer(std::vector<double> values, std::vector<double> targets) const
	{
		std::vector<double> nodesLossesDer;

		if (outAct == ActFunc::Softmax && lossFunc == LossFunc::CrossEntropy)
		{
			for (int i = 0; i < values.size(); i++)
				nodesLossesDer.push_back(values[i] - targets[i]);
		}
		else if (outAct == ActFunc::Sigmoid && lossFunc == LossFunc::SquaredError)
		{
			for (int i = 0; i < values.size(); i++)
			{
				double sigm = 1.f / (1 + exp(-values[i]));
				nodesLossesDer.push_back(sigm * (1 - sigm) * 2 * (values[i] - targets[i]));
			}
		}
		else std::exit(-101);

		return nodesLossesDer;
	}
}