#include "math.hpp"

namespace Mlib {
	int random(int v1, int v2)
	{
		int min = (v1 < v2) ? v1 : v2;
		int max = (v1 > v2) ? v1 : v2;

		static std::random_device dev;
		static std::mt19937 rng(dev());
	
		std::uniform_int_distribution<std::mt19937::result_type> dist6(0, max - min);
		return (dist6(rng) + min);
	}
	
	double hypotenuse(double cat1, double cat2) {
		return sqrt(cat1 * cat1 + cat2 * cat2);
	}
}