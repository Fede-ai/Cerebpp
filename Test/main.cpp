#include <iostream>
#include "Mlib/Utility/time.hpp"
#include "Mlib/System/system.hpp"

int main()
{
	auto t = Mlib::getTime();
	while (true) {
		std::cout << int(t.asSec()) << "\n";
		Mlib::sleep(Mlib::Seconds(1));
		t = Mlib::getTime();
	}
	
	return 0;
}