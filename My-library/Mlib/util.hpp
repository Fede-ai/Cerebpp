#pragma once
#include "Util/dataPoint.hpp"

namespace Mlib {
	//return time in milliseconds
	size_t getTime();
	//sleep for x milliseconds
	void sleep(int milliseconds);
	//get a random integet in a given range
	int random(int min, int max);
}