#include <iostream>
#include <thread>
#include "NN/nn.hpp"
#include <SFML/Graphics.hpp>

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

	Mlib::NN ai("best");
	if (ai.sizes.size() == 0) {
		ai = Mlib::NN({ 1, 60, 60, 1 }, Mlib::NN::ReLU, Mlib::NN::Sigmoid, Mlib::NN::SquaredError, true);
		std::cout << "created new nn from scratch\n";
	}
	else
		std::cout << "loaded nn from best.txt\n";

	bool stop = false;
	std::thread thrd([&stop]() {
		std::string ok;
		std::cin >> ok;
		stop = true;
	});

	Mlib::Batch batch(dataset);
	int i = 0;
	while (true) {
		ai.train(batch, 0.4f, 0.2f);
		std::cout << ++i << " - " << ai.loss(batch) << "\n";

		if (stop)
			break;
	}

	ai.save("best");
	thrd.join();

	double mean = 0;
	for (const auto& d : dataset.datapoints)
		mean += d.target[0] * 2.5 - 0.5;
	mean /= dataset.datapoints.size();

	//compute r^2 values for the regressions
	double m = 0.015036575, q = -5.060441735;
	double sst = 0, linearSrr = 0, aiSrr = 0;
	for (const auto& d : dataset.datapoints) {
		double x = (d.data[0] + 3) * 100, y = d.target[0] * 2.5 - 0.5;
		sst += std::pow(y - mean, 2);

		linearSrr += std::pow(y - (x * m + q), 2);

		double aiPred = ai.computePrediction(d)[0] * 2.5 - 0.5;
		aiSrr += std::pow(y - aiPred, 2);

		//std::cout << y << " -> " << (x * m + q) << " - " << aiPred << "\n";
	}

	std::cout << "\nmean = " << mean << "\n";
	std::cout << "sst = " << sst << "\n";
	std::cout << "R^2 linear regression = " << 1 - linearSrr / sst << ", srr = " << linearSrr << "\n";
	std::cout << "R^2 ai model = " << 1 - aiSrr / sst << ", srr = " << aiSrr << "\n";

	int sX = 315, eX = 435;
	float sY = -0.6f, eY = 2;
	int sizeX = 1200, sizeY = 780;

	sf::Image img;
	img.create(sizeX, sizeY, sf::Color(220, 220, 220));

	//draw horizontal grid
	for (float y = sY + 0.2f; y < eY; y += 0.2f) {
		float yPos = sizeY - (y - sY) / (eY - sY) * sizeY;
		for (int x = 0; x < sizeX; x++)
			img.setPixel(x, std::floor(yPos), sf::Color(120, 120, 120));
	}

	//draw vertical grid
	for (int x = sX + 12; x < eX; x += 12) {
		float xPos = float(x - sX) / (eX - sX) * sizeX;
		for (int y = 0; y < sizeY; y++)
			img.setPixel(std::floor(xPos), y, sf::Color(120, 120, 120));
	}

	//draw all datapoints
	for (const auto& d : dataset.datapoints) {
		float xVal = (d.data[0] + 3) * 100;
		float yVal = d.target[0] * 2.5f - 0.5f;

		float xPos = (xVal - sX) / (eX - sX) * sizeX;
		float yPos = sizeY - (yVal - sY) / (eY - sY) * sizeY;

		for (int a = -1; a < 3; a++)
			for (int b = -1; b < 3; b++)
				img.setPixel(std::floor(xPos) + a, std::floor(yPos) + b, sf::Color::Magenta);
	}

	//draw both regressions
	for (int x = 0; x < sizeX; x++) {
		float xVal = (x / float(sizeX)) * (eX - sX) + sX;

		float yVal = float(xVal * m + q);
		float yPos = sizeY - (yVal - sY) / (eY - sY) * sizeY;
		img.setPixel(x, std::floor(yPos), sf::Color::Red);
		img.setPixel(x, std::floor(yPos) + 1, sf::Color::Red);
		img.setPixel(x, std::floor(yPos) + 2, sf::Color::Red);

		yVal = ai.computePrediction({ xVal / 100 - 3 })[0] * 2.5f - 0.5f;
		yPos = sizeY - (yVal - sY) / (eY - sY) * sizeY;
		img.setPixel(x, std::floor(yPos), sf::Color::Blue);
		img.setPixel(x, std::floor(yPos) + 1, sf::Color::Blue);
		img.setPixel(x, std::floor(yPos) + 2, sf::Color::Blue);
	}

	img.saveToFile("img.png");

	return 0;
}