#pragma once
#include <vector>
#include "util.hpp"
#include <fstream>
#include <functional>
#include <string>

namespace Mlib {
	struct Datapoint
	{
		//setup the datapoint with the given data, target adn id values
		Datapoint(std::vector<double> inData, std::vector<double> inTarget, int inId)
			:
			data(inData), target(inTarget), id(inId)
		{
		}
		//create an empty datapoint
		Datapoint() {}
		//load datapoint from string
		Datapoint(std::string str) {}

		std::vector<double> data;
		std::vector<double> target;
		int id = NULL;
	};

	struct Dataset
	{
		Dataset(std::function<Datapoint(std::string)> func, std::string path, int num, bool labels = false)
		{
			//skip n lines
			/*for (int i = 0; i < 0; ++i) {
				file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
			}*/

			std::ifstream file(path);
			if (labels)
				file.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); //skip labels row
			for (int n = 0; n < num; n++)
			{
				std::string line;
				getline(file, line);
				data.push_back(func(line));
			}
				
			file.close();
		}
		Dataset() {}

		std::vector<Datapoint> data;
	};

	struct Batch
	{
		Batch(Dataset d, int size)
		{
			for (int i = 0; i < size; i++)
			{
				int index = Mlib::random(0, d.data.size() - 1);
				data.push_back(d.data[index]);
				d.data.erase(d.data.begin() + index);
			}
		}
		Batch() {}

		std::vector<std::reference_wrapper<Datapoint>> data;
	};
}