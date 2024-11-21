#include "Crb/System/file.hpp"
//#include <Windows.h>

namespace Crb {

	std::string getOpenFilePath(const char types[]) {
		/*OPENFILENAMEA file;
		CHAR path[100];
	
		ZeroMemory(&file, sizeof(file));
		file.lStructSize = sizeof(file);
		file.hwndOwner = NULL;
		file.lpstrFile = path;
		file.lpstrFile[0] = '\0';
		file.nMaxFile = sizeof(path);
		file.lpstrFilter = types;
		file.nFilterIndex = 1;
		file.lpstrFileTitle = NULL;
		file.nMaxFileTitle = 0;
		file.lpstrInitialDir = NULL;
		file.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST;
	
		GetOpenFileNameA(&file);
		return path;*/
		return "";
	}
	
	std::string getSaveFilePath(const char types[]) {
		/*OPENFILENAMEA file;
		CHAR path[100];
	
		ZeroMemory(&file, sizeof(file));
		file.lStructSize = sizeof(file);
		file.hwndOwner = NULL;
		file.lpstrFilter = types;
		file.lpstrFile = path;
		file.lpstrFile[0] = '\0';
		file.nMaxFile = sizeof(path);
		file.Flags = OFN_EXPLORER | OFN_FILEMUSTEXIST | OFN_HIDEREADONLY;
		file.lpstrDefExt = (LPCSTR)"txt";
	
		GetSaveFileNameA(&file);
		return path;*/
		return "";
	}
}