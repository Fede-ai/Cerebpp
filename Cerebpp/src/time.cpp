#include "Crb/Utility/time.hpp"
#include <thread>
#include <chrono>
 
namespace Crb {
	double Time::asSec() const {
		return value / 1000.0;
	}
	size_t Time::asMil() const {
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

	Time seconds(double seconds) 
	{
		Time t;
		t.value = static_cast<size_t>(seconds * 1000);
		return t;
	}
	Time milliseconds(size_t milliseconds) 
	{
		Time t;
		t.value = milliseconds;
		return t;
	}
	Time currentTime()
	{
		Time t;
		t.value = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
		return t;
	}

	void Clock::start()
	{
		if (running)
			return;

		running = true;
		started = currentTime();
	}
	void Clock::stop()
	{
		if (!running)
			return;

		running = false;
		elapsed += (currentTime() - started);
	}
	void Clock::setTo(const Time t)
	{
		if (running)
			started = currentTime();
		elapsed = t;
	}
	Time Clock::getTime() const
	{
		if (running)
			return (elapsed + (currentTime() - started));
		else
			return elapsed;
	}
	bool Clock::isRunning() const
	{
		return running;
	}

	void sleep(Time time) {
		std::this_thread::sleep_for(std::chrono::milliseconds(time.asMil()));
	}
}
