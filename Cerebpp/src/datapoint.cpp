#include "Crb/Utility/datapoint.hpp"
#include <iostream>
#include <algorithm>
#include <random>

namespace Mlib
{
	Datapoint::Datapoint(std::vector<float> inData, std::vector<float> inTarget, int inId)
		:
		data(inData), target(inTarget), id(inId)
	{
	}
	Datapoint::Datapoint()
	{
	}

	Dataset::Dataset(std::function<Datapoint(std::string)> func, std::string path, bool labels)
	{
		//skip n lines
		/*for (int i = 0; i < 0; ++i) {
			file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
		}*/

		std::ifstream file(path);
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
	Dataset::Dataset()
	{
	}

	size_t Dataset::size() const
	{
		return datapoints.size();
	}

	void Dataset::loadFromFile(std::function<Datapoint(std::string)> func, std::string path, bool labels)
	{
		*this = Dataset(func, path, labels);
	}

	Batch::Batch(const Dataset& dataset, int n)
	{
		if (dataset.datapoints.size() < n) {
			std::cout << "ERROR: dataset size: " << dataset.datapoints.size() << ", batch size: " << n;
			std::exit(-104);
		}

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

	size_t Batch::size() const
	{
		return datapoints.size();
	}
}