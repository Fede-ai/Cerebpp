#include "Crb/FNN/layer.hpp"
#include <fstream>
#include <sstream>

namespace Crb {
	FNN& FNN::operator=(FNN other)
	{
		sizes = other.sizes;
		hidAct = other.hidAct;
		outAct = other.outAct;
		lossFunc = other.lossFunc;

		for (int i = 0; i < other.layers.size(); i++)
			layers.push_back(new Layer(*other.layers[i]));

		return *this;
	}

	FNN::FNN(std::vector<int> inSizes, ActFunc inHidAct, ActFunc inOutAct, LossFunc inLossFunc, bool rand)
		:
		hidAct(inHidAct),
		outAct(inOutAct),
		lossFunc(inLossFunc),
		sizes(inSizes)
	{
		for (int layer = 1; layer < sizes.size(); layer++)
			layers.push_back(new Layer(sizes[layer - 1], sizes[layer], rand, inHidAct, inOutAct, inLossFunc));
	}
	FNN::FNN(std::string path)
	{
		loadFromFile(path);
	}
	FNN::~FNN()
	{
		for (int i = 0; i < layers.size(); i++)
			delete layers[i];
	}

	std::vector<float> FNN::feedforward(const std::vector<float>& data) 
	{
		//pass the input through the first layer
		std::vector<float> values = layers[0]->forwardPass(data);

		//pass the values through the rest of the layers
		for (int layer = 1; layer < layers.size() - 1; layer++)
			values = layers[layer]->forwardPass(values);

		//pass the values through the output layer
		values = layers[layers.size() - 1]->forwardPass(values, true);

		return values;
	}
	std::vector<float> FNN::feedforward(const Datapoint& datapoint)
	{
		return feedforward(datapoint.data);
	}
	void FNN::backProp(const Datapoint& datapoint)
	{
		feedforward(datapoint);
		std::vector<float> nodeValues = layers[layers.size() - 1]->computeOutputNodeValues(datapoint.target);
		layers[layers.size() - 1]->updateGradients(nodeValues);

		for (int layer = static_cast<int>(layers.size()) - 2; layer >= 0; layer--) {
			nodeValues = layers[layer]->computeHiddenNodeValues(nodeValues, *layers[layer + 1]);
			layers[layer]->updateGradients(nodeValues);
		}
	}
	float FNN::loss(const Batch& batch)
	{
		float loss = 0;
		for (const auto& d : batch.datapoints)
			loss += layers[layers.size() - 1]->loss(feedforward(d), d.get().target);
		return (loss / batch.datapoints.size());
	}

	void FNN::backPropagation(const Batch& batch, float learnRate, float momentum)
	{
		for (auto& d : batch.datapoints)
			backProp(d.get());

		for (auto& layer : layers)
			layer->applyGradients(learnRate, momentum, static_cast<int>(batch.size()));

		for (auto& layer : layers)
			layer->clearGradients();
	}

	void FNN::loadFromFile(std::string path)
	{
		std::ifstream file;
		std::string line, token;
		file.open(path, std::ios::in);
		if (!file.is_open())
			throw std::runtime_error("couldn't open the file to load the fnn");

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

		for (int i = 1; i < sizes.size(); i++) {
			getline(file, line);
			layers.push_back(new Layer(sizes[i - 1], sizes[i], hidAct, outAct, lossFunc, line));
		}

		file.close();
	}
	void FNN::save(std::string path) const
	{
		std::ofstream file;
		file.open(path, std::ios::out | std::ios::trunc);

		if (!file.is_open())
			throw std::runtime_error("couldn't open the file to save the fnn");

		//write size
		for (auto size : sizes)
			file << size << ',';

		file << '\n' << static_cast<int>(hidAct) << ',' << static_cast<int>(outAct) 
			<< ',' << static_cast<int>(lossFunc) << ",\n";

		//write biases and weights
		for (const auto& layer : layers)
			file << layer->toString() << '\n';

		file.close();
	}
}