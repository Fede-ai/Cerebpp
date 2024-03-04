#pragma once
#include <chrono>

namespace Mlib {
	class Time {
	public:		
		//default constructor, time set to 0
		Time();

		double asSec();
		size_t asMil();
			
		friend Time seconds(double seconds);
		friend Time milliseconds(size_t milliseconds);

		bool operator==(const Time o) const;
		bool operator!=(const Time o) const;
		bool operator>(const Time o) const;
		bool operator>=(const Time o) const;
		bool operator<(const Time o) const;
		bool operator<=(const Time o) const;

		Time operator+=(const Time o);
		Time operator+(const Time o) const;
		Time operator-=(const Time o);
		Time operator-(const Time o) const;
		Time operator*=(const double k);
		Time operator*(const double k) const;
		Time operator/=(const double k);
		Time operator/(const double k) const;

	private:
		size_t value = 0; //in milliseconds
	};
	
	Time seconds(double seconds);
	Time milliseconds(size_t milliseconds);
}