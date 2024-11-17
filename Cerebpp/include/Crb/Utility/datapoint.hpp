#pragma once
#include <vector>
#include <fstream>
#include <functional>
#include <string>

namespace Crb {
	struct Datapoint
	{
		//create an empty datapoint
		Datapoint() = default;
		//setup the datapoint with the given data, target and id values
		Datapoint(int inId, std::vector<float> inData, std::vector<float> inTarget);

		std::vector<float> data;
		std::vector<float> target;
		int id = NULL;
	};

	struct Dataset
	{
		//create an empty dataset
		Dataset() = default;
		//load a dataset from a file (.txt or .csv files should be used).
		//each line of the file will be converted to a datapoint using the 'func' parameter
		Dataset(std::function<Datapoint(std::string)> func, std::string path, bool labels = true);

		//load a dataset from a file(.txt or .csv files should be used).
		//each line of the file will be converted to a datapoint using the 'func' parameter
		void loadFromFile(std::function<Datapoint(std::string)> func, std::string path, bool labels = true);
		//move 'spitPercentage' percent datapoints in the 'other' dataset
		void split(Dataset& other, float spitPercentage);

		size_t size() const;

		//a vector of the datapoints that form the dataset
		std::vector<Datapoint> datapoints;
	};

	struct Batch
	{
		//create an empty batch
		Batch() = default;
		//create a batch taking 'n' random datapoints from the given dataset
		Batch(const Dataset& dataset, int n);
		//create a batch containing all the datapoints in the dataset
		Batch(const Dataset& dataset);
		//load 'n' random datapoints from the given dataset
		void loadFromDataset(const Dataset& dataset, int n);
		//load all the datapoints in the dataset
		void loadFromDataset(const Dataset& dataset);

		size_t size() const;

		//a vector of references to datapoints
		std::vector<std::reference_wrapper<const Datapoint>> datapoints;
	};
}