#pragma once
#include "Util/io.hpp"

namespace Mlib {
	//return time in milliseconds
	size_t getTime();
	//sleep for x milliseconds
	void sleep(int milliseconds);
	//get a random integer in a given range
	int random(int min, int max);
}