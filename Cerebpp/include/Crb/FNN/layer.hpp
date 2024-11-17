#include "Crb/FNN/fnn.hpp"

namespace Crb {
	class FNN::Layer {
	public:
		/*create a new layer from size, function.parameters are eigher set to 0 ('rand' = false)
		or to a random number between -1 and 1 ('rand' = true)*/
		Layer(int inNumBef, int inNumAft, bool rand, ActFunc inHidAct, ActFunc inOutAct, LossFunc inLossFunc);
		//load layer from size, functions and a string
		Layer(int inNumBef, int inNumAft, ActFunc inHidAct, ActFunc inOutAct, LossFunc inLossFunc, std::string string);
		std::vector<float> forwardPass(const std::vector<float>& inputs, bool output = false);
		//used for backpropagation
		std::vector<float> computeHiddenNodeValues(const std::vector<float>& nodeValuesAft, const Layer& layerAft) const;
		//used for backpropagation
		std::vector<float> computeOutputNodeValues(const std::vector<float>& targets) const;

		//update the gradients 
		void updateGradients(const std::vector<float>& nodeValues);
		//update all parameters applying the gradients
		void applyGradients(float learnRate, float momentum, int batchSize);
		//set all gradients to 0
		void clearGradients();

		//calculate the losses for each value-target pair
		float loss(const std::vector<float>& values, const std::vector<float>& targets) const;

		//save the layer to a string format
		std::string toString() const;

	private:
		std::vector<float> hiddenAct(const std::vector<float>& values) const;
		std::vector<float> hiddenActDer(const std::vector<float>& values) const;
		std::vector<float> outputAct(const std::vector<float>& values) const;

		ActFunc hidAct = NoAct, outAct = NoAct;
		LossFunc lossFunc = NoLoss;

		int numBef = -1, numAft = -1;
		//relative to the nodes after
		std::vector<float> biases;
		//access a specific weight value. notation: weights[nodeBef][nodeAft]
		std::vector<std::vector<float>> weights;

		//relative to the nodes after
		std::vector<float> biasesGradients, biasesVelocities;
		//notation: weightsGradients[nodeBef][nodeAft]
		std::vector<std::vector<float>> weightsGradients, weightsVelocities;

		//the activatedValues of the layer before, one for each node bef
		std::vector<float> inputValues;
		//weighted sum of the activatedValues of the layer before, one for each node aft
		std::vector<float> weightedValues;
		//weightedValues after passing through the activation function, one for each node aft
		std::vector<float> activatedValues;
	};
}
