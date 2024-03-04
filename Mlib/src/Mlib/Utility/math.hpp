#pragma once
#include <random>

namespace Mlib {
	//return a random signed integer between 'v1' and 'v2' (both signed integers)
	int random(int v1, int v2);

	//return the hypotenuse of a right triangle, given the two legs
	double hypotenuse(double cat1, double cat2);
}