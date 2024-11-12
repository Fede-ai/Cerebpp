#include "datapoint.hpp"
#include <iostream>
#include <algorithm>

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
			data.push_back(func(line));

			if (file.eof())
				break;
		}

		file.close();
	}
	Dataset::Dataset()
	{
	}

	void Dataset::loadFromFile(std::function<Datapoint(std::string)> func, std::string path, bool labels)
	{
		*this = Dataset(func, path, labels);
	}

	Batch::Batch(Dataset& dataset, int n)
	{
		if (dataset.data.size() < n) {
			std::cout << "ERROR: dataset size: " << dataset.data.size() << ", batch size: " << n;
			std::exit(-104);
		}

		static std::random_device rd;
		static std::mt19937 gen(rd());
		std::sample(dataset.data.begin(), dataset.data.end(), 
			std::back_inserter(data), n, gen);
	}
	Batch::Batch()
	{
	}
}