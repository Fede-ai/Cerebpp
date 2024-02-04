#pragma once
#include "../Data/vec2.hpp"

namespace Mlib
{
	//doc: https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-setcursorpos
	bool setMousePos(Vec2i pos);

	//https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-getasynckeystate
	//https://learn.microsoft.com/en-us/windows/win32/api/winuser/ns-winuser-input
}