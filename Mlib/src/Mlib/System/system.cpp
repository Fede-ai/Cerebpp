#include "system.hpp"

namespace Mlib {
	void sleep(Time time) {
		std::this_thread::sleep_for(std::chrono::milliseconds(time.asMil()));
	}

	Vec2i displaySize() {
		return Vec2i(GetSystemMetrics(SM_CXSCREEN), GetSystemMetrics(SM_CYSCREEN));
	}
}