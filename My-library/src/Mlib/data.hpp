#pragma once
#include <vector>
#include "util.hpp"
#include <fstream>
#include <functional>
#include <string>

namespace Mlib {
	struct Datapoint
	{
		//setup the datapoint with the given data, target and id values
		Datapoint(std::vector<double> inData, std::vector<double> inTarget, int inId)
			:
			data(inData), target(inTarget), id(inId)
		{
		}
		//create an empty datapoint
		Datapoint() {}

		std::vector<double> data;
		std::vector<double> target;
		int id = NULL;
	};

	struct Dataset
	{
		/*load a dataset from a file(.txt or .csv files should be used).
		each line of the file will be converted to a datapoint using the 'func' parameter*/
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
		//create an empty dataset
		Dataset() {}

		//a vector of the datapoints that form the dataset
		std::vector<Datapoint> data;
	};

	struct Batch
	{
		//create a batch taking 'n' random datapoints from the given dataset
		Batch(Dataset dataset, int n)
		{
			for (int i = 0; i < n; i++)
			{
				int index = Mlib::random(0, dataset.data.size() - 1);
				data.push_back(dataset.data[index]);
				dataset.data.erase(dataset.data.begin() + index);
			}
		}
		//create an empty batch
		Batch() {}

		//a vector of references to datapoints
		std::vector<std::reference_wrapper<Datapoint>> data;
	};
}