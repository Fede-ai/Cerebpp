#include <iostream>
#include "Mlib/Util/util.hpp"
#include "Mlib/Io/keyboard.hpp"

int main()
{
	while (true)
	{
		std::cout << Mlib::Keyboard::getAsyncState(Mlib::Keyboard::A) << '\n';
		Mlib::sleep(20);
	}

	return 0;
}