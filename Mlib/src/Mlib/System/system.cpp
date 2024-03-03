#include "system.hpp"

namespace Mlib {
	Time getTime() {
		return milliseconds(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count());
	}

	void sleep(Time time) {
		std::this_thread::sleep_for(std::chrono::milliseconds(time.asMil()));
	}

	Vec2i displaySize() {
		return Vec2i(GetSystemMetrics(SM_CXSCREEN), GetSystemMetrics(SM_CYSCREEN));
	}
}