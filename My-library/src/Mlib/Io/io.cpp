#include "io.hpp"
#include <Windows.h>

namespace Mlib
{
	bool setMousePos(Vec2i pos)
	{
		return SetCursorPos(pos.x, pos.y);
	}

}