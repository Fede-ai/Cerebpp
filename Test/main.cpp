#include <iostream>
#include "Mlib/System/file.hpp"

int main()
{
	std::cout << Mlib::getSaveFilePath("Test (*.prova)\0*.prova\0Text (*.txt)\0*.txt\0");
	
	return 0;
}