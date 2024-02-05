#pragma once

namespace Mlib
{
	template <typename T> struct Vec2
	{
		Vec2() {};
		Vec2(T inX, T inY) : x(inX), y(inY) {};
		
		//casting operator
		template<typename A> operator Vec2<A>() const {
			return Vec2<A>(static_cast<A>(x), static_cast<A>(y));
		}

		//addition operators
		template<typename A> Vec2<T> operator+(const Vec2<A> o) const {
			return Vec2<T>(x + o.x, y + o.y);
		}
		template<typename A> Vec2<T>& operator+=(const Vec2<A> o) {
			x += static_cast<T>(o.x);
			y += static_cast<T>(o.y);
			return *this;
		}
		//subtraction operators
		template<typename A> Vec2<T> operator-(const Vec2<A> o) const {
			return Vec2<T>(x - o.x, y - o.y);
		}
		template<typename A> Vec2<T>& operator-=(const Vec2<A> o) {
			x -= static_cast<T>(o.x);
			y -= static_cast<T>(o.y);
			return *this;
		}
		//multiplication operators
		template<typename A> Vec2<T> operator*(const Vec2<A> o) const {
			return Vec2<T>(x * o.x, y * o.y);
		}
		template<typename A> Vec2<T>& operator*=(const Vec2<A> o) {
			x *= static_cast<T>(o.x);
			y *= static_cast<T>(o.y);
			return *this;
		}
		//division operators
		template<typename A> Vec2<T> operator/(const Vec2<A> o) const {
			return Vec2<T>(x / o.x, y / o.y);
		}
		template<typename A> Vec2<T>& operator/=(const Vec2<A> o) {
			x /= static_cast<T>(o.x);
			y /= static_cast<T>(o.y);
			return *this;
		}

		T x;
		T y;
	};

	typedef Vec2<int> Vec2i;
	typedef Vec2<float> Vec2f;
}