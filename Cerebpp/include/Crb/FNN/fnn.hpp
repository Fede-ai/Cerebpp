#pragma once
#include "Crb/Utility/datapoint.hpp"

namespace Crb 
{
	class FNN {
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
		FNN(std::vector<int> inSizes, ActFunc inHidAct, ActFunc inOutAct, LossFunc inLossFunc, bool rand = false);
		//load the ai parameters from a .txt file (do not include the file extension in the path parameter)
		FNN(std::string path);

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
		
		class Layer;
		std::vector<Layer*> layers;

		ActFunc hidAct = NoAct, outAct = NoAct;
		LossFunc lossFunc = NoLoss;
	};
}