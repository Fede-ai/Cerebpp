#pragma once
#include <random>

namespace Mlib {
	//return a random integer between 'min' adn 'max'
	int random(int min, int max);

	//return the hypotenuse of a right triangle, given the two legs
	float hypotenuse(float cat1, float cat2);
}