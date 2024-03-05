#pragma once
#include "../Utility/vec2.hpp"
#include "../Utility/time.hpp"
#include <thread>
#include <wtypes.h>

namespace Mlib {
	//sleep for a given amount of time
	void sleep(Time time);

	//get the size of the main display
	Vec2i displaySize();
}