#pragma once
#include <chrono>

namespace Mlib {
	class Time {
	public:		
		//default constructor, time set to 0
		Time() : value(0) {}

		double asSec();
		size_t asMil();

		friend Time seconds(double seconds);
		friend Time milliseconds(size_t milliseconds);

	private:
		size_t value = 0; //in milliseconds
	};
	
	Time seconds(double seconds);
	Time milliseconds(size_t milliseconds);
}