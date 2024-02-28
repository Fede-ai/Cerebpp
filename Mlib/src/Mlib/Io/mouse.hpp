#pragma once
#include "../Data/vec2.hpp"

namespace Mlib
{
	namespace Mouse
	{
		enum Button {
			Left = 0x01,
			Middle = 0x04,
			Right = 0x02,
			Side1 = 0x05,
			Side2 = 0x06
		};

		//doc: https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-setcursorpos
		void setPos(Vec2i pos, bool relative = false);
		/*doc: https://learn.microsoft.com/en-us/windows/win32/api/winuser/ns-winuser-mouseinput
		set the state of a specific button to be up or down*/
		void setState(Button but, bool down);
		//calls 2 times 'setState', pressing and releasing a button
		void simulateClick(Button but);	
		void simulateScroll(int scrollValue);

		//doc: https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-getcursorpos
		Vec2i getPos();
		//doc: https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-getkeystate
		bool isButPressed(Button but);
		//doc: https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-getkeystate
		bool isButToggled(Button but);
	}
}