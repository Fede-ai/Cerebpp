#include <iostream>
#include <thread>
#include "NN/nn.hpp"

static Mlib::Datapoint readData(std::string str) {
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
			dp.target.push_back((stof(piece) + 0.5f) / 2.5f);

		str = str.substr(str.find_first_of(',') + 1);
	}

	return dp;
}

int main()
{
	Mlib::Dataset dataset;
	dataset.loadFromFile(readData, "C:/Users/feder/Desktop/two-var/dataset.csv");

	Mlib::NN ai({ 1, 60, 60, 1 }, Mlib::NN::Sigmoid, Mlib::NN::Sigmoid, Mlib::NN::SquaredError, true);
	Mlib::Batch batch(dataset, 609);

	bool stop = false;
	std::thread thrd([&stop]() {
		std::string ok;
		std::cin >> ok;
		stop = true;
	});

	int i = 0;
	while (true) {
		ai.train(batch, 0.5f, 0.2f);
		std::cout << ++i << " - " << ai.loss(batch) << "\n";

		if (stop)
			break;
	}

	ai.save("best");
	thrd.join();

	double mean = 0;
	for (const auto& d : dataset.data)
		mean += d.target[0] * 2.5 - 0.5;
	mean /= dataset.data.size();

	std::cout << "mean = " << mean << "\n";

	double m = 0.015036575, q = -5.060441735;
	double sst = 0, linearSrr = 0, aiSrr = 0;
	for (const auto& d : dataset.data) {
		double x = (d.data[0] + 3) * 100, y = d.target[0] * 2.5 - 0.5;
		sst += std::pow(y - mean, 2);

		linearSrr += std::pow(y - (x * m + q), 2);

		double aiPred = ai.computePrediction(d)[0] * 2.5 - 0.5;
		aiSrr += std::pow(y - aiPred, 2);

		std::cout << y << " -> " << (x * m + q) << " - " << aiPred << "\n";
	}

	std::cout << "sst = " << sst << "\n";
	std::cout << "R^2 linear regression = " << 1 - linearSrr / sst << ", srr = " << linearSrr << "\n";
	std::cout << "R^2 ai model = " << 1 - aiSrr / sst << ", srr = " << aiSrr << "\n";

	return 0;
}