#include "mouse.hpp"
#include <Windows.h>
#include "../Util/util.hpp"

namespace Mlib
{
	namespace Mouse
	{
		void setState(Button but, bool down)
		{
			INPUT in;
			in.type = INPUT_MOUSE;
			in.mi.time = 0;
			in.mi.mouseData = 0;

			if (but == Button::Left)
			{
				if (down)
					in.mi.dwFlags = MOUSEEVENTF_LEFTDOWN;
				else
					in.mi.dwFlags = MOUSEEVENTF_LEFTUP;
			}
			else if (but == Button::Middle)
			{
				if (down)
					in.mi.dwFlags = MOUSEEVENTF_MIDDLEDOWN;
				else
					in.mi.dwFlags = MOUSEEVENTF_MIDDLEUP;
			}
			else if (but == Button::Right)
			{
				if (down)
					in.mi.dwFlags = MOUSEEVENTF_RIGHTDOWN;
				else
					in.mi.dwFlags = MOUSEEVENTF_RIGHTUP;
			}
			else if (but == Button::Side1 || but == Button::Side2)
			{
				if (down)
					in.mi.dwFlags = MOUSEEVENTF_XDOWN;
				else
					in.mi.dwFlags = MOUSEEVENTF_XUP;

				if (but == Button::Side1)
					in.mi.mouseData = XBUTTON1;
				else
					in.mi.mouseData = XBUTTON2;
			}

			SendInput(1, &in, sizeof(INPUT));
		}
		void setPos(Vec2i pos, bool relative)
		{
			Vec2i targ = pos;
			if (relative)
				targ = getPos() + pos;

			SetCursorPos(targ.x, targ.y);
		}	
		void simulateClick(Button but)
		{
			setState(but, true);
			setState(but, false);
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

		Vec2i getPos()
		{
			POINT pos;
			GetCursorPos(&pos);
			return Vec2i(pos.x, pos.y);
		}
		bool isButPressed(Button but)
		{
			return GetKeyState(but) < 0;
		}
		bool isButToggled(Button but)
		{
			return GetKeyState(but) & 0x01;
		}
	}
}