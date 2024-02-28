#pragma once
#include <vector>
#include <fstream>
#include <functional>
#include <string>
#include "../Util/util.hpp"

namespace Mlib {
	struct Datapoint
	{
		//setup the datapoint with the given data, target and id values
		Datapoint(std::vector<float> inData, std::vector<float> inTarget, int inId);
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
		Dataset(std::function<Datapoint(std::string)> func, std::string path, int num, bool labels = false);
		//create an empty dataset
		Dataset() {}

		//a vector of the datapoints that form the dataset
		std::vector<Datapoint> data;
	};

	struct Batch
	{
		//create a batch taking 'n' random datapoints from the given dataset
		Batch(Dataset& dataset, int n);
		//create an empty batch
		Batch() {}

		//a vector of references to datapoints
		std::vector<std::reference_wrapper<Datapoint>> data;
	};
}