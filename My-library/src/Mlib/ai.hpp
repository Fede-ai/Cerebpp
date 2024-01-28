#pragma once
#include <vector>
#include <string>
#include "data.hpp"

namespace Mlib {
	namespace Func {
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
	};

	class Ai {
	public:
		/*creare a new ai with the given size and function. the 'rand' parameter determinates whether the
		weights and biases should be initialized as 0 or as a random number between -1 and 1*/
		Ai(std::vector<int> inSizes, Func::ActFunc inHidAct, Func::ActFunc inOutAct, Func::LossFunc inLossFunc, bool rand = false);
		//load the ai parameters from a .txt file (do not include the file extension in the path parameter)
		Ai(std::string path);

		//compute the predicted target values for a given datapoint
		std::vector<double> computePrediction(Datapoint datapoint);
		//calculate the average loos across the given batch
		double loss(Batch batch);
		//train the ai on a given batch
		void train(Batch batch, double learnRate, double momentum);
		//save the ai parameters in a .txt file (do not include the file extension in the path parameter)
		void save(std::string path) const;

		//a vector containing the number of nodes in each layer
		std::vector<int> sizes;

	private:	
		void backProp(Datapoint datapoint);
		
		class Layer {
		public:
			/*create a new layer from size, function.parameters are eigher set to 0 ('rand' = false)
			or to a random number between -1 and 1 ('rand' = true)*/
			Layer(int inNumBef, int inNumAft, bool rand, Func::ActFunc inHidAct, Func::ActFunc inOutAct, Func::LossFunc inLossFunc);
			//load layer from size, functions and a string
			Layer(int inNumBef, int inNumAft, Func::ActFunc inHidAct, Func::ActFunc inOutAct, Func::LossFunc inLossFunc, std::string string);
			std::vector<double> computeHidden(std::vector<double> inputs);
			//used for backpropagation
			std::vector<double> computeHiddenNodeValues(std::vector<double> nodeValuesAfter, Layer layerAft) const;
			std::vector<double> computeOutput(std::vector<double> inputs);
			//used for backpropagation
			std::vector<double> computeOutputNodeValues(std::vector<double> targets) const;

			//update the gradients 
			void updateGradients(std::vector<double> nodeValues);
			//update all parameters applying the gradients
			void applyGradients(double learnRate, double momentum, int batchSize);
			//set all gradients to 0
			void clearGradients();

			//calculate the losses for each value-target pair
			double loss(std::vector<double> values, std::vector<double> targets) const;

			//save the layer to a string format
			std::string toString() const;

		private:
			std::vector<double> hiddenAct(std::vector<double> values) const;
			std::vector<double> hiddenActDer(std::vector<double> values) const;
			std::vector<double> outputAct(std::vector<double> values) const;

			std::vector<double> lossAndOutputActDer(std::vector<double> values, std::vector<double> targets) const;
			Func::ActFunc hidAct = Func::NoAct, outAct = Func::NoAct;
			Func::LossFunc lossFunc = Func::NoLoss;

			int numBef = -1, numAft = -1;
			//relative to the nodes after
			std::vector<double> biases;
			//access a specific weight value. notation: weights[nodeBef][nodeAft]
			std::vector<std::vector<double>> weights;

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

		std::vector<Layer> layers;
		Func::ActFunc hidAct = Func::NoAct, outAct = Func::NoAct;
		Func::LossFunc lossFunc = Func::NoLoss;
	};
}