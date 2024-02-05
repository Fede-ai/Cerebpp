#pragma once
#include "../Data/vec2.hpp"

namespace Mlib
{
	namespace Mouse
	{
		enum MouseButton {
			None = -1,
			Left = 0x01,
			Middle = 0x04,
			Right = 0x02,
			Side1 = 0x05,
			Side2 = 0x06
		};

		//doc: https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-setcursorpos
		void setMousePos(Vec2i pos, bool relative = false);
		//doc: https://learn.microsoft.com/en-us/windows/win32/api/winuser/ns-winuser-mouseinput
		void setButState(MouseButton but, bool down);
		//calls 2 times 'setButState', pressing and releasing a button
		void simulateClick(MouseButton but);	
		void simulateScroll(int scrollValue);

		//doc: https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-getcursorpos
		Vec2i getMousePos();
		//doc: https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-getkeystate
		bool isButPressed(MouseButton but);
		bool isButToggled(MouseButton but);
	}
}