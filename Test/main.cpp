#include <iostream>
#include "NN/nn.hpp"

Mlib::Datapoint readData(std::string str) {
	Mlib::Datapoint dp;

	for (int i = 0; i < 5; i++) {
		std::string piece = str.substr(0, str.find_first_of(','));

		if (i == 0)
			dp.id = stoi(piece) * 100;
		else if (i == 1)
			dp.id += stoi(piece);
		else if (i == 3)
			dp.data.push_back(stof(piece) / 100 - 3);
		else if (i == 4)
			dp.target.push_back((stof(piece) + 0.5) / 2.5);

		str = str.substr(str.find_first_of(',') + 1);
	}

	return dp;
}

int main()
{
	Mlib::Dataset dataset;
	dataset.loadFromFile(readData, "C:/Users/feder/Desktop/two-var/dataset.csv");

	Mlib::NN ai({ 1, 20, 20, 1 }, Mlib::NN::Sigmoid, Mlib::NN::Sigmoid, Mlib::NN::SquaredError, true);

	while (true) {
		Mlib::Batch batch(dataset, 609);
		ai.train(batch, 0.01, 0.1);
		std::cout << ai.loss(batch) << "\n";
	}

	//for (auto& d : dataset.data)
	//	std::cout << d.id << " - " << d.data[0] << ", " << d.target[0] << "\n";

	return 0;
}