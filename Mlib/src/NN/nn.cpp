#include "nn.hpp"
#include <Windows.h>

namespace Mlib {
	NN::NN(std::vector<int> inSizes, ActFunc inHidAct, ActFunc inOutAct, LossFunc inLossFunc, bool rand)
		:
		hidAct(inHidAct),
		outAct(inOutAct),
		lossFunc(inLossFunc),
		sizes(inSizes)
	{
		for (int layer = 1; layer < sizes.size(); layer++)
			layers.push_back(Layer(sizes[layer - 1], sizes[layer], rand, inHidAct, inOutAct, inLossFunc));
	}
	NN::NN(std::string path)
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
		hidAct = static_cast<ActFunc>(stoi(token));
		getline(funcStream, token, ',');
		outAct = static_cast<ActFunc>(stoi(token));
		getline(funcStream, token, ',');
		lossFunc = static_cast<LossFunc>(stoi(token));

		for (int i = 1; i < sizes.size(); i++)
		{
			getline(file, line);
			layers.push_back(Layer(sizes[i - 1], sizes[i], hidAct, outAct, lossFunc, line));
		}

		file.close();
	}

	std::vector<float> NN::computePrediction(const std::vector<float>& data) 
	{
		//pass the input through the first layer
		std::vector<float> values = layers[0].forwardPass(data);

		//pass the values through the rest of the layers
		for (int layer = 1; layer < layers.size() - 1; layer++)
			values = layers[layer].forwardPass(values);

		//pass the values through the output layer
		values = layers[layers.size() - 1].forwardPass(values, true);

		return values;
	}
	std::vector<float> NN::computePrediction(const Datapoint& datapoint)
	{
		return computePrediction(datapoint.data);
	}
	void NN::backProp(const Datapoint& datapoint)
	{
		computePrediction(datapoint);
		std::vector<float> nodeValues = layers[layers.size() - 1].computeOutputNodeValues(datapoint.target);
		layers[layers.size() - 1].updateGradients(nodeValues);

		for (int layer = static_cast<int>(layers.size()) - 2; layer >= 0; layer--)
		{
			nodeValues = layers[layer].computeHiddenNodeValues(nodeValues, layers[layer + 1]);
			layers[layer].updateGradients(nodeValues);
		}
	}
	float NN::loss(const Batch& batch)
	{
		float loss = 0;
		for (const auto& d : batch.datapoints)
			loss += layers[0].loss(computePrediction(d), d.get().target);
		return (loss / batch.datapoints.size());
	}

	void NN::train(const Batch& batch, float learnRate, float momentum)
	{
		for (auto& d : batch.datapoints)
			backProp(d.get());

		for (auto& layer : layers)
			layer.applyGradients(learnRate, momentum, static_cast<int>(batch.size()));

		for (auto& layer : layers)
			layer.clearGradients();
	}
	void NN::save(std::string path) const
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
		for (const auto& layer : layers)
			file << layer.toString() << '\n';

		file.close();
	}
}