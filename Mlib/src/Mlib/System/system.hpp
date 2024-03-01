#include "../Utility/vec2.hpp"
#include <chrono>
#include <thread>
#include <wtypes.h>

namespace Mlib {
	//return time in milliseconds
	size_t getTime() {
		return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
	}

	//sleep for x milliseconds
	void sleep(int milliseconds) {
		std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds));
	}

	//get the size of the main display
	Vec2i displaySize() {
		return Vec2i(GetSystemMetrics(SM_CXSCREEN), GetSystemMetrics(SM_CYSCREEN));
	}
}