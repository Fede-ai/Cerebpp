#include "datapoint.hpp"
#include <iostream>

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
		std::vector<int> index;
		for (int i = 0; i < dataset.data.size(); i++)
			index.push_back(i);

		for (int i = 0; i < n; i++)
		{
			int num = Mlib::random(0, static_cast<int>(index.size() - 1));
			data.push_back(std::ref(dataset.data[num]));
			index.erase(index.begin() + num);
		}
	}
	Batch::Batch()
	{
	}
}