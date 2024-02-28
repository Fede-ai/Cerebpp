#include <iostream>
#include "Mlib/Io/keyboard.hpp"
#include "Mlib/Util/util.hpp"

int main()
{
	while (true)
	{
		std::cout << Mlib::Keyboard::getAsyncState(Mlib::Keyboard::A) << '\n';
		Mlib::sleep(10);
	}

	return 0;
}