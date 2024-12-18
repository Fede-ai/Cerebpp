#pragma once
#include <string>

namespace Crb
{
	namespace Keyboard
	{
		//virtual key codes (3 -> 254) for evey key
		enum Key {
			Cancel = 0x03,
			Backspace = 0x08,
			Tab = 0x09,
			Clear = 0x0C,
			Enter = 0x0D,
			Shift = 0x10,
			Ctrl = 0x11,
			Alt = 0x12,
			Pause = 0x13,
			CapsLock = 0x14,
			Esc = 0x1B,
			Space = 0x20,
			PageUp = 0x21,
			PageDown = 0x22,
			End = 0x23,
			Home = 0x24,
			Left = 0x25,
			Up = 0x26,
			Right = 0x27,
			Down = 0x28,
			Select = 0x29,
			Print = 0x2A,
			Execute = 0x2B,
			Snapshot = 0x2C, //PRINT SCREEN key
			Insert = 0x2D, //INS key
			Delete = 0x2E,
			Help = 0x2F,
			Num0 = 0x30,
			Num1 = 0x31,
			Num2 = 0x32,
			Num3 = 0x33,
			Num4 = 0x34,
			Num5 = 0x35,
			Num6 = 0x36,
			Num7 = 0x37,
			Num8 = 0x38,
			Num9 = 0x39,
			A = 0x41,
			B = 0x42,
			C = 0x43,
			D = 0x44,
			E = 0x45,
			F = 0x46,
			G = 0x47,
			H = 0x48,
			I = 0x49,
			J = 0x4A,
			K = 0x4B,
			L = 0x4C,
			M = 0x4D,
			N = 0x4E,
			O = 0x4F,
			P = 0x50,
			Q = 0x51,
			R = 0x52,
			S = 0x53,
			T = 0x54,
			U = 0x55,
			V = 0x56,
			W = 0x57,
			X = 0x58,
			Y = 0x59,
			Z = 0x5A,
			LWin = 0x5B, //left Windows key
			RWin = 0x5C, //right Windows key
			Apps = 0x5D, //applications key
			Sleep = 0x5F,
			Numpad0 = 0x60,
			Numpad1 = 0x61,
			Numpad2 = 0x62,
			Numpad3 = 0x63,
			Numpad4 = 0x64,
			Numpad5 = 0x65,
			Numpad6 = 0x66,
			Numpad7 = 0x67,
			Numpad8 = 0x68,
			Numpad9 = 0x69,
			Multiply = 0x6A,
			Add = 0x6B,
			Separator = 0x6C,
			Subtract = 0x6D,
			Decimal = 0x6E,
			Divide = 0x6F,
			F1 = 0x70,
			F2 = 0x71,
			F3 = 0x72,
			F4 = 0x73,
			F5 = 0x74,
			F6 = 0x75,
			F7 = 0x76,
			F8 = 0x77,
			F9 = 0x78,
			F10 = 0x79,
			F11 = 0x7A,
			F12 = 0x7B,
			F13 = 0x7C,
			F14 = 0x7D,
			F15 = 0x7E,
			F16 = 0x7F,
			F17 = 0x80,
			F18 = 0x81,
			F19 = 0x82,
			F20 = 0x83,
			F21 = 0x84,
			F22 = 0x85,
			F23 = 0x86,
			F24 = 0x87,
			NumLock = 0x90,
			ScrollLock = 0x91,
			LShift = 0xA0,
			RShift = 0xA1,
			LCtrl = 0xA2,
			RCtrl = 0xA3,
			LAlt = 0xA4,
			RAlt = 0xA5,
			BrowserBack = 0xA6,
			BrowserForward = 0xA7,
			BrowserRefresh = 0xA8,
			BrowserStop = 0xA9,
			BrowserSearch = 0xAA,
			BrowserFavorites = 0xAB,
			BrowserHome = 0xAC,
			Mute = 0xAD,
			VolumeDown = 0xAE,
			VolumeUp = 0xAF,
			NextTrack = 0xB0,
			PrevTrack = 0xB1,
			Stop = 0xB2,
			PlayPause = 0xB3,
			LaunchMail = 0xB4,
			LaunchMediaSelect = 0xB5,
			LaunchApp1 = 0xB6,
			LaunchApp2 = 0xB7,
			OEM_1 = 0xBA, //it can vary by keyboard. for the US standard keyboard, the ;: key
			Plus = 0xBB,
			Comma = 0xBC,
			Minus = 0xBD,
			Period = 0xBE,
			OEM_2 = 0xBF, //it can vary by keyboard. for the US standard keyboard, the /? key
			OEM_3 = 0xC0, //it can vary by keyboard. for the US standard keyboard, the `~ key
			OEM_4 = 0xDB, //it can vary by keyboard. for the US standard keyboard, the [{ key
			OEM_5 = 0xDC, //it can vary by keyboard. for the US standard keyboard, the \| key
			OEM_6 = 0xDD, //it can vary by keyboard. for the US standard keyboard, the ]} key
			OEM_7 = 0xDE, //it can vary by keyboard. for the US standard keyboard, the '" key
			OEM_8 = 0xDF, //it can vary by keyboard
			OEM_102 = 0xE2, //the <> keys on the US standard keyboard, or the \| key on the non US key keyboard
			EraseEOF = 0xF9,
			Play = 0xFA,
			Zoom = 0xFB,
			OEM_Clear = 0xFE
		};

		/*doc: https://learn.microsoft.com/en-us/windows/win32/api/winuser/ns-winuser-keybdinput
		set the state of a specific key to be up or down*/
		void setState(Key key, bool down);
		//calls 2 times 'setState', pressing and releasing a button
		void simulateStroke(Key key);
		//simulates writing each char in a string, doesnt trigger any shift/alt/ctrl presses
		void writeWord(std::string word);
		
		/*get whether a key is currently pressed (true) or not (false)
		doc: https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-getkeystate */
		bool isKeyPressed(Key key);
		/*get whether a key is currently toggled (true) or not (false)
		//doc: https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-getkeystate */
		bool isKeyToggled(Key key);
		/*get wheater a key has been pressed since last call
		https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-getasynckeystate */
		bool getAsyncState(Key key);
	}
}