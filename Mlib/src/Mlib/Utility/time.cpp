#include "time.hpp"
 
namespace Mlib {
	Time::Time()
		: 
		value(0) 
	{}

	double Time::asSec() {
		return value / 1000.0;
	}
	size_t Time::asMil() {
		return value;
	}

	bool Time::operator==(const Time o) const {
		return (value == o.value);
	}
	bool Time::operator!=(const Time o) const {
		return (value != o.value);
	}
	bool Time::operator>(const Time o) const {
		return (value > o.value);
	}
	bool Time::operator>=(const Time o) const {
		return (value >= o.value);
	}
	bool Time::operator<(const Time o) const {
		return (value < o.value);
	}
	bool Time::operator<=(const Time o) const {
		return (value <= o.value);
	}

	Time Time::operator+=(const Time o) {
		value += o.value;
		return *this;
	}
	Time Time::operator+(const Time o) const {
		Time t;
		t.value = value + o.value;
		return t;
	}
	Time Time::operator-=(const Time o) {
		value -= o.value;
		return *this;
	}
	Time Time::operator-(const Time o) const {
		Time t;
		t.value = value - o.value;
		return t;
	}
	Time Time::operator*=(const double k) {
		value = static_cast<size_t>(value * k);
		return *this;
	}
	Time Time::operator*(const double k) const {
		Time t;
		t.value = static_cast<size_t>(value * k);
		return t;
	}
	Time Time::operator/=(const double k) {
		value = static_cast<size_t>(value / k);
		return *this;
	}
	Time Time::operator/(const double k) const {
		Time t;
		t.value = static_cast<size_t>(value / k);
		return t;
	}

	Time seconds(double seconds) {
		Time t;
		t.value = static_cast<size_t>(seconds * 1000);
		return t;
	}
	Time milliseconds(size_t milliseconds) {
		Time t;
		t.value = milliseconds;
		return t;
	}
}
