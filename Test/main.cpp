#include <iostream>
#include "Utility/time.hpp"
#include "System/system.hpp"

void take(Mlib::Clock& c) {
	while (true) {
		std::string cmd;
		std::cin >> cmd;

		if (cmd[0] == 'r')
			c.setTo(Mlib::Time());
		else if (cmd[0] == 's')
			c.start();
		else if (cmd[0] == 'e')
			c.stop();
	}
}

int main()
{
	Mlib::Clock cl;
	cl.start();

	std::thread th(take, std::ref(cl));

	while (true) {
		std::cout << cl.getTime().asSec() << "\n";
		Mlib::sleep(Mlib::milliseconds(100));
	}

	return 0;
}