#pragma once 
#include <string>

namespace Crb {
	//open the file select dialog and select a file
	//types = the valid file extentions. The following format MUST be used: 
	//"All Files (*.*)\0*.*\0Text (*.txt)\0*.txt\0"
	std::string getOpenFilePath(const char types[]);

	//open the file save dialog and select a path and file name
	//types = the valid file extentions. The following format MUST be used: 
	//"All Files (*.*)\0*.*\0Text (*.txt)\0*.txt\0"
	std::string getSaveFilePath(const char types[]);
}