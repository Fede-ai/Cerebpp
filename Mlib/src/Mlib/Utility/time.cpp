#include "time.hpp"
 
namespace Mlib {
	double Time::asSec() {
		return value / 1000.0;
	}
	size_t Time::asMil() {
		return value;
	}

	Time seconds(double seconds) {
		Time t;
		t.value = seconds * 1000;
		return t;
	}
	Time milliseconds(size_t milliseconds) {
		Time t;
		t.value = milliseconds;
		return t;
	}
}
