#include "Crb/Utility/datapoint.hpp"
#include <random>
#include <fstream>
#include <numeric>

namespace Crb
{
	Datapoint::Datapoint(int inId, std::vector<float> inData, std::vector<float> inTarget)
		:
		data(inData), target(inTarget), id(inId)
	{
	}

	Dataset::Dataset(std::function<Datapoint(std::string)> func, std::string path, bool labels)
	{
		//skip n lines
		/*for (int i = 0; i < 0; ++i) {
			file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
		}*/

		std::ifstream file(path);
		if (!file.is_open())
			throw std::runtime_error("invalid dataset file path");

		if (labels)
			file.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); //skip labels row

		while (true) {
			std::string line;
			getline(file, line);
			datapoints.push_back(func(line));

			if (file.eof())
				break;
		}

		file.close();
	}

	void Dataset::loadFromFile(std::function<Datapoint(std::string)> func, std::string path, bool labels)
	{
		*this = Dataset(func, path, labels);
	}
	void Dataset::split(Dataset& other, float spitPercentage)
	{
		if (spitPercentage <= 0 || spitPercentage >= 1)
			throw std::runtime_error("invalid split percentage");

		int n = static_cast<int>(std::floor(datapoints.size() * spitPercentage));

		std::vector<int> indexes(datapoints.size());
		std::iota(std::begin(indexes), std::end(indexes), 0);

		std::vector<int> indexesToMove;
		static std::random_device dev;
		static std::mt19937 rng(dev());
		std::sample(indexes.begin(), indexes.end(),
			std::back_inserter(indexesToMove), n, rng);

		for (int i = int(indexesToMove.size()) - 1; i >= 0; i--) {
			int index = indexesToMove[i];
			other.datapoints.push_back(datapoints[index]);
			datapoints.erase(datapoints.begin() + index);
		}
	}

	size_t Dataset::size() const
	{
		return datapoints.size();
	}

	Batch::Batch(const Dataset& dataset, int n)
	{
		if (dataset.datapoints.size() < n)
			throw std::runtime_error("cannot create a batch larger than the dataset");

		static std::random_device dev;
		static std::mt19937 rng(dev());
		std::sample(dataset.datapoints.begin(), dataset.datapoints.end(), 
			std::back_inserter(datapoints), n, rng);
	}
	Batch::Batch(const Dataset& dataset)
	{
		for (const auto& d : dataset.datapoints)
			datapoints.push_back(d);
	}

	void Batch::loadFromDataset(const Dataset& dataset, int n)
	{
		*this = Batch(dataset, n);
	}
	void Batch::loadFromDataset(const Dataset& dataset)
	{
		*this = Batch(dataset);
	}

	size_t Batch::size() const
	{
		return datapoints.size();
	}
}