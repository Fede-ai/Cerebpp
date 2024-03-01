#pragma once
#include <random>

namespace Mlib {
	//return a random integer between 'min' adn 'max'
	int random(int min, int max)
	{
		static std::random_device dev;
		static std::mt19937 rng(dev());

		std::uniform_int_distribution<std::mt19937::result_type> dist6(min, max);
		return dist6(rng);
	}

	//return the hypotenuse of a right triangle, given the two legs
	float hypotenuse(float cat1, float cat2) {
		return sqrt(cat1*cat1 + cat2*cat2);
	}
}