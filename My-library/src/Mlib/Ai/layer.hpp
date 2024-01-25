#pragma once
#include <vector>

namespace Mlib {
	enum ActFunc {
		Sigmoid = 0,
		ReLU = 1,
		Softmax = 2
	};

	enum LossFunc {
		SquaredError = 0,
		CrossEntropy = 1
	};

	class Layer {
	public:
		Layer(int inNumBef, int inNumAft, bool rand, ActFunc inHidAct, ActFunc inOutAct, LossFunc inLossFunc);
		std::vector<double> computeHidden(std::vector<double> inputs);
		std::vector<double> computeHiddenNodeValues(std::vector<double> nodeValuesAfter, Layer layerAft) const;
		std::vector<double> computeOutput(std::vector<double> inputs);
		std::vector<double> computeOutputNodeValues(std::vector<double> targets) const;

		void updateGradients(std::vector<double> nodeValues);
		void applyGradients(double learnRate, double momentum, int batchSize);
		void clearGradients();

		double loss(std::vector<double> values, std::vector<double> targets) const;

		const int numBef, numAft;	
		//relative to the nodes after
		std::vector<double> biases;	
		//notation: weights[nodeBef][nodeAft]
		std::vector<std::vector<double>> weights;

	private:
		std::vector<double> hiddenAct(std::vector<double> values) const;
		std::vector<double> hiddenActDer(std::vector<double> values) const;
		std::vector<double> outputAct(std::vector<double> values) const;

		std::vector<double> lossAndOutputActDer(std::vector<double> values, std::vector<double> targets) const;
		ActFunc hidAct, outAct; 
		LossFunc lossFunc;

		//relative to the nodes after
		std::vector<double> biasesGradients, biasesVelocities;
		//notation: weightsGradients[nodeBef][nodeAft]
		std::vector<std::vector<double>> weightsGradients, weightsVelocities;

		//the activatedValues of the layer before, one for each node bef
		std::vector<double> inputValues;
		//weighted sum of the activatedValues of the layer before, one for each node aft
		std::vector<double> weightedValues;
		//weightedValues after passing through the activation function, one for each node aft
		std::vector<double> activatedValues;
	};
}
