#pragma once
#include "../Utility/vec2.hpp"
#include "../Utility/time.hpp"
#include <thread>
#include <wtypes.h>

namespace Mlib {
	//get the current system time
	Time getTime() {
		return Milliseconds(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count());
	}

	//sleep for a given amount of time
	void sleep(Time time) {
		std::this_thread::sleep_for(std::chrono::milliseconds(time.asMil()));
	}

	//get the size of the main display
	Vec2i displaySize() {
		return Vec2i(GetSystemMetrics(SM_CXSCREEN), GetSystemMetrics(SM_CYSCREEN));
	}
}