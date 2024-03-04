#include <iostream>
#include "Mlib/Utility/time.hpp"
#include "Mlib/System/system.hpp"

int main()
{
	Mlib::Vec2f ok(32.2, 2.13);

	std::cout << (ok + Mlib::Vec2<char>(21, 2)).x;

	return 0;
}