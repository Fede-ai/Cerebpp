#pragma once
#include <vector>
#include <fstream>
#include <functional>
#include <string>
#include "../Utility/math.hpp"

namespace Mlib {
	struct Datapoint
	{
		//setup the datapoint with the given data, target and id values
		Datapoint(std::vector<float> inData, std::vector<float> inTarget, int inId);
		//create an empty datapoint
		Datapoint();

		std::vector<float> data;
		std::vector<float> target;
		int id = NULL;
	};

	struct Dataset
	{
		/*load a dataset from a file(.txt or .csv files should be used).
		each line of the file will be converted to a datapoint using the 'func' parameter*/
		Dataset(std::function<Datapoint(std::string)> func, std::string path, bool labels = true);
		//create an empty dataset
		Dataset();

		size_t size() const;
		void loadFromFile(std::function<Datapoint(std::string)> func, std::string path, bool labels = true);

		//a vector of the datapoints that form the dataset
		std::vector<Datapoint> datapoints;
	};

	struct Batch
	{
		//create a batch taking 'n' random datapoints from the given dataset
		Batch(const Dataset& dataset, int n);
		//create a batch containing all the datapoints in the dataset
		Batch(const Dataset& dataset);
		//create an empty batch
		Batch() = default;

		size_t size() const;

		//a vector of references to datapoints
		std::vector<std::reference_wrapper<const Datapoint>> datapoints;
	};
}