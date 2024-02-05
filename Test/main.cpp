#include <iostream>
#include "Mlib/Util/util.hpp"
#include "Mlib/Io/mouse.hpp"

int main()
{
	Mlib::setMousePos({ 100, 0 }, true);
	Mlib::setMouseState(Mlib::MouseButton::Right, true);
	Mlib::setMouseState(Mlib::MouseButton::Right, false);
	return 0;
}