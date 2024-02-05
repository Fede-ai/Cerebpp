#pragma once
#include "../Data/vec2.hpp"

namespace Mlib
{
	enum MouseButton {
		None = -1,
		Left = 0,
		Middle = 1,
		Right = 2,
		Side1 = 3,
		Side2 = 4
	};

	//doc: https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-setcursorpos
	void setMousePos(Vec2i pos, bool relative = false);
	//doc: https://learn.microsoft.com/en-us/windows/win32/api/winuser/ns-winuser-mouseinput
	void setMouseState(MouseButton but, bool down);
	//calls 2 times 'setMouseState', pressing and releasing a button
	void simulateClick(MouseButton but);	
	void simulateScroll(int scrollValue);

	//doc: https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-getcursorpos
	Vec2i getMousePos();

}