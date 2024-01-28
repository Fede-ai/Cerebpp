#include "../util.hpp"
#include <chrono>
#include <thread>
#include <random>

namespace Mlib {
	size_t getTime()
	{
		return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
	}

	void sleep(int milliseconds)
	{
		std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds));
	}

	int random(int min, int max)
	{
		static std::random_device dev;
		static std::mt19937 rng(dev());

		std::uniform_int_distribution<std::mt19937::result_type> dist6(min, max);
		return dist6(rng);
	}
}