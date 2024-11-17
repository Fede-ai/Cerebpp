#pragma once
#include <vector>
#include <fstream>
#include <sstream>
#include "../Utility/datapoint.hpp"

namespace Crb 
{
	class NN {
	public:	
		//activation functions for hidden and output layers
		enum ActFunc {
			NoAct = -1,
			Sigmoid = 0,
			//cannot be output activation function
			ReLU = 1,
			Softmax = 2
		};
		//loss function for output layer
		enum LossFunc {
			NoLoss = -1,
			SquaredError = 0,
			CrossEntropy = 1
		};

		/*creare a new ai with the given size and function. the 'rand' parameter determinates whether the
		weights and biases should be initialized as 0 or as a random number between -1 and 1*/
		NN(std::vector<int> inSizes, ActFunc inHidAct, ActFunc inOutAct, LossFunc inLossFunc, bool rand = false);
		//load the ai parameters from a .txt file (do not include the file extension in the path parameter)
		NN(std::string path);

		//compute the predicted target values for a given input data
		std::vector<float> computePrediction(const std::vector<float>& data);
		//compute the predicted target values for a given datapoint
		std::vector<float> computePrediction(const Datapoint& datapoint);
		//calculate the average loss across the given batch
		float loss(const Batch& batch);
		//train the ai on a given batch
		void train(const Batch& batch, float learnRate, float momentum);
		//save the ai parameters in a .txt file (do not include the file extension in the path parameter)
		void save(std::string path) const;

		//a vector containing the number of nodes in each layer
		std::vector<int> sizes;

	private:	
		void backProp(const Datapoint& datapoint);
		
		class Layer {
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

		std::vector<Layer> layers;

		ActFunc hidAct = NoAct, outAct = NoAct;
		LossFunc lossFunc = NoLoss;
	};
}