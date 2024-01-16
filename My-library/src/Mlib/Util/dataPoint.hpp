#pragma once
#include <vector>

namespace Mlib {
	struct DataPoint
	{
		//setup the datapoint with the given data, target adn id values
		DataPoint(std::vector<double> inData, std::vector<double> inTarget, int inId)
			: 
			data(inData), target(inTarget), id(inId)
		{
		}
		//create an empty datapoint
		DataPoint() {};

		std::vector<double> data;	
		std::vector<double> target;
		int id = NULL;
	};
}