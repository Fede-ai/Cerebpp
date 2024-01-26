#include "../ai.hpp"
#include <Windows.h>
#include <fstream>
#include <sstream>

namespace Mlib {
	Ai::Ai(std::vector<int> inSizes, Func::ActFunc inHidAct, Func::ActFunc inOutAct, Func::LossFunc inLossFunc, bool rand)
		:
		hidAct(inHidAct),
		outAct(inOutAct),
		lossFunc(inLossFunc)
	{
		sizes = inSizes;

		for (int layer = 1; layer < sizes.size(); layer++)
			layers.push_back(Layer(sizes[layer - 1], sizes[layer], rand, inHidAct, inOutAct, inLossFunc));
	}
	Ai::Ai(std::string path)
	{
		std::fstream file;
		std::string line, token;
		file.open(path + ".txt", std::ios::in);
		if (!file.is_open())
			return;

		//extract and load layers sizes
		getline(file, line);
		std::istringstream sizesStream(line);
		while (getline(sizesStream, token, ','))
			sizes.push_back(stoi(token));
		//extract and load functions used
		getline(file, line);
		std::istringstream funcStream(line);
		getline(funcStream, token, ',');
		hidAct = static_cast<Func::ActFunc>(stoi(token));
		getline(funcStream, token, ',');
		outAct = static_cast<Func::ActFunc>(stoi(token));
		getline(funcStream, token, ',');
		lossFunc = static_cast<Func::LossFunc>(stoi(token));

		for (int i = 1; i < sizes.size(); i++)
		{
			getline(file, line);
			layers.push_back(Layer(sizes[i - 1], sizes[i], hidAct, outAct, lossFunc, line));
		}

		file.close();
	}

	std::vector<double> Ai::computePrediction(Datapoint datapoint)
	{
		std::vector<double> dataDouble;
		for (auto d : datapoint.data)
			dataDouble.push_back(d/255.f);

		std::vector<double> values = layers[0].computeHidden(dataDouble);
		for (int layer = 1; layer < layers.size() - 1; layer++)
			values = layers[layer].computeHidden(values);
		values = layers[layers.size() - 1].computeOutput(values);

		return values;
	}
	void Ai::backProp(Datapoint datapoint)
	{
		computePrediction(datapoint);
		std::vector<double> nodeValues = layers[layers.size() - 1].computeOutputNodeValues(datapoint.target);
		layers[layers.size() - 1].updateGradients(nodeValues);

		for (int layer = layers.size() - 2; layer >= 0; layer--)
		{
			nodeValues = layers[layer].computeHiddenNodeValues(nodeValues, layers[layer + 1]);
			layers[layer].updateGradients(nodeValues);
		}
	}
	double Ai::loss(Batch batch)
	{
		double loss = 0;
		for (auto d : batch.data)
			loss += layers[0].loss(computePrediction(d), d.get().target);
		return (loss / batch.data.size());
	}

	void Ai::train(Batch batch, double learnRate, double momentum)
	{
		for (auto& d : batch.data)
			backProp(d.get());

		for (auto& layer : layers)
			layer.applyGradients(learnRate, momentum, batch.data.size());

		for (auto& layer : layers)
			layer.clearGradients();
	}
	void Ai::save(std::string path) const
	{
		std::fstream file;
		file.open(path + ".txt", std::ios::out | std::ios::trunc);

		if (!file.is_open())
			return;

		//write size
		for (auto size : sizes)
			file << size << ',';
		file << '\n';

		file << static_cast<int>(hidAct) << ',' << static_cast<int>(outAct) << ',' << static_cast<int>(lossFunc) << ",\n";

		//write biases and weights
		for (auto layer : layers)
		{
			file << layer.toString();

			file << '\n';
		}

		file.close();
	}
}