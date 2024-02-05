#include "mouse.hpp"
#include <Windows.h>
#include "../Util/util.hpp"

namespace Mlib
{
	namespace Mouse
	{
		void setButState(MouseButton but, bool down)
		{
			INPUT in;
			in.type = INPUT_MOUSE;
			in.mi.time = 0;
			in.mi.mouseData = 0;

			if (but == MouseButton::Left)
			{
				if (down)
					in.mi.dwFlags = MOUSEEVENTF_LEFTDOWN;
				else
					in.mi.dwFlags = MOUSEEVENTF_LEFTUP;
			}
			else if (but == MouseButton::Middle)
			{
				if (down)
					in.mi.dwFlags = MOUSEEVENTF_MIDDLEDOWN;
				else
					in.mi.dwFlags = MOUSEEVENTF_MIDDLEUP;
			}
			else if (but == MouseButton::Right)
			{
				if (down)
					in.mi.dwFlags = MOUSEEVENTF_RIGHTDOWN;
				else
					in.mi.dwFlags = MOUSEEVENTF_RIGHTUP;
			}
			else if (but == MouseButton::Side1 || but == MouseButton::Side2)
			{
				if (down)
					in.mi.dwFlags = MOUSEEVENTF_XDOWN;
				else
					in.mi.dwFlags = MOUSEEVENTF_XUP;

				if (but == MouseButton::Side1)
					in.mi.mouseData = XBUTTON1;
				else
					in.mi.mouseData = XBUTTON2;
			}

			SendInput(1, &in, sizeof(INPUT));
		}
		void setMousePos(Vec2i pos, bool relative)
		{
			Vec2i targ = pos;
			if (relative)
				targ = getMousePos() + pos;

			SetCursorPos(targ.x, targ.y);
		}	
		void simulateClick(MouseButton but)
		{
			setButState(but, true);
			setButState(but, false);
		}
		void simulateScroll(int scrollValue)
		{
			INPUT input;
			input.type = INPUT_MOUSE;
			input.mi.dx = 0;
			input.mi.dy = 0;
			input.mi.mouseData = scrollValue;
			input.mi.dwFlags = MOUSEEVENTF_WHEEL;
			input.mi.time = 0;
			input.mi.dwExtraInfo = 0;

			SendInput(1, &input, sizeof(INPUT));
		}

		Vec2i getMousePos()
		{
			POINT pos;
			GetCursorPos(&pos);
			return Vec2i(pos.x, pos.y);
		}
		bool isButPressed(MouseButton but)
		{
			return GetKeyState(but) < 0;
		}
		bool isButToggled(MouseButton but)
		{
			return GetKeyState(but) & 0x01;
		}
	}
}