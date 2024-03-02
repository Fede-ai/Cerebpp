#pragma once
#include <chrono>

namespace Mlib {
	class Time {
	public:		
		//default constructor, time set to 0
		Time() : value(0) {}

		double asSec() {
			return value / 1000.0;
		}
		size_t asMil() {
			return value;
		}

		friend Time Seconds(double seconds);
		friend Time Milliseconds(size_t milliseconds);

	private:
		size_t value = 0; //in milliseconds
	};
	
	Time Seconds(double seconds) {
		Time t;
		t.value = seconds * 1000;
		return t;
	}
	Time Milliseconds(size_t milliseconds) {
		Time t;
		t.value = milliseconds;
		return t;
	}
}