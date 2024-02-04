#pragma once

namespace Mlib
{
	template <typename T> struct Vec2
	{
		Vec2(T inX, T inY) : x(inX), y(inY) {}
		
		template<typename A> operator Vec2<A>() const {
			return Vec2<A>(static_cast<A>(x), static_cast<A>(y));
		}

		T x;
		T y;
	};

	typedef Vec2<char> Vec2c;
	typedef Vec2<int> Vec2i;
	typedef Vec2<float> Vec2f;
	typedef Vec2<double> Vec2d;
}