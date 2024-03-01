#pragma once
#include <vector>
#include <fstream>
#include <sstream>
#include "../Utility/datapoint.hpp"

namespace Mlib 
{
	class Ai {
	public:	
		//activation functions for hidden and output layers
		enum ActFunc {
			NoAct = -1,
			Sigmoid = 0,
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
		Ai(std::vector<int> inSizes, ActFunc inHidAct, ActFunc inOutAct, LossFunc inLossFunc, bool rand = false);
		//load the ai parameters from a .txt file (do not include the file extension in the path parameter)
		Ai(std::string path);

		//compute the predicted target values for a given datapoint
		std::vector<float> computePrediction(Datapoint datapoint);
		//calculate the average loos across the given batch
		float loss(Batch batch);
		//train the ai on a given batch
		void train(Batch batch, float learnRate, float momentum);
		//save the ai parameters in a .txt file (do not include the file extension in the path parameter)
		void save(std::string path) const;

		//a vector containing the number of nodes in each layer
		std::vector<int> sizes;

	private:	
		void backProp(Datapoint datapoint);
		
		class Layer;
		std::vector<Layer> layers;

		ActFunc hidAct = NoAct, outAct = NoAct;
		LossFunc lossFunc = NoLoss;
	};
}