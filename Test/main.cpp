#include <iostream>
#include "Mlib/Util/util.hpp"
#include "Mlib/Io/keyboard.hpp"
#include "Mlib/Ai/ai.hpp"

int main()
{
	Mlib::sleep(3000);
	Mlib::Keyboard::writeWord("nice");
	return 0;
}