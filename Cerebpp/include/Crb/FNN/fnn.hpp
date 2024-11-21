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

		//create empty fnn
		FNN() = default;
		FNN& operator=(FNN other);

		//creare a new fnn with the given size and function. the 'rand' parameter determinates whether the
		//weights and biases should be initialized as 0 or as a random number between -1 and 1
		FNN(std::vector<int> inSizes, ActFunc inHidAct, ActFunc inOutAct, LossFunc inLossFunc, bool rand = false);
		//load the fnn parameters from a file (usually a .txt)
		FNN(std::string path);
		~FNN();

		//compute the predicted target values for a given input data
		std::vector<float> feedforward(const std::vector<float>& data);
		//compute the predicted target values for a given datapoint
		std::vector<float> feedforward(const Datapoint& datapoint);
		//calculate the average loss across the given batch
		float loss(const Batch& batch);
		//train the fnn on a given batch
		void backPropagation(const Batch& batch, float learnRate, float momentum);

		void loadFromFile(std::string path);
		//save the fnn parameters to a file (.txt is recommended)
		void save(std::string path) const;

	private:	
		void backProp(const Datapoint& datapoint);

		class Layer;
		std::vector<Layer*> layers;
		std::vector<int> sizes;

		ActFunc hidAct = NoAct, outAct = NoAct;
		LossFunc lossFunc = NoLoss;
	};
}