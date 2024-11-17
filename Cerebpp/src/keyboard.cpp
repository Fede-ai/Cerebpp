#include "Crb/System/keyboard.hpp"
#include <Windows.h>

namespace Crb
{	
	namespace Keyboard
	{	
		void setState(Key key, bool down)
		{
			INPUT input;
			input.type = INPUT_KEYBOARD;
			input.ki.wVk = key;
			if (down)
				input.ki.dwFlags = 0;
			else
				input.ki.dwFlags = KEYEVENTF_KEYUP;

			SendInput(1, &input, sizeof(INPUT));
		}
		void simulateStroke(Key key)
		{
			setState(key, true);
			setState(key, false);
		}
		void writeWord(std::string word)
		{
			for (auto c : word)
			{
				simulateStroke(Key(VkKeyScan(c)));
			}
		}

		bool isKeyPressed(Key key)
		{
			return GetKeyState(key) < 0;
		}
		bool isKeyToggled(Key key)
		{
			return GetKeyState(key) & 0x01;
		}
		bool getAsyncState(Key key)
		{
			return GetAsyncKeyState(key) & 0x01;
		}
	}
}