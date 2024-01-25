#pragma once
#include <vector>
#include <string>
#include "data.hpp"
#include "Ai/layer.hpp"

namespace Mlib {
	class Ai {
	public:
		Ai(std::vector<int> inSizes, ActFunc inHidAct, ActFunc inOutAct, LossFunc inLossFunc, bool rand = false);
		Ai(std::string path);
		std::vector<double> forwardProp(Datapoint datapoint);
		double loss(std::vector<Datapoint> datapoints);
		void learn(std::vector<Datapoint> datapoints, double learnRate, double momentum);
		void save() const;

	private:	
		void backProp(Datapoint datapoint);
		std::vector<int> sizes;
		std::vector<Layer> layers;
		ActFunc hidAct, outAct;
		LossFunc lossFunc;
	};
}