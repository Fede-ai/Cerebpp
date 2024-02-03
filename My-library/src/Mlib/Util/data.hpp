#pragma once
#include <vector>
#include <fstream>
#include <functional>
#include <string>
#include "../util.hpp"

namespace Mlib {
	struct Datapoint
	{
		//setup the datapoint with the given data, target and id values
		Datapoint(std::vector<float> inData, std::vector<float> inTarget, int inId)
			:
			data(inData), target(inTarget), id(inId)
		{
		}
		//create an empty datapoint
		Datapoint() {}

		std::vector<float> data;
		std::vector<float> target;
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
		Batch(Dataset& dataset, int n)
		{
			if (dataset.data.size() < n) 
				std::exit(-104);
			std::vector<int> index;
			for (int i = 0; i < dataset.data.size(); i++)
				index.push_back(i);

			for (int i = 0; i < n; i++)
			{
				int num = Mlib::random(0, index.size() - 1);
				data.push_back(std::ref(dataset.data[num]));
				index.erase(index.begin() + num);
			}
		}
		//create an empty batch
		Batch() {}

		//a vector of references to datapoints
		std::vector<std::reference_wrapper<Datapoint>> data;
	};
}