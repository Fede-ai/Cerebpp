#pragma once

namespace Crb {
	//definitions of this class' functions must be inline 
	//and in the header because of templates rules

	template <typename T> struct Vec2
	{
		Vec2() {}
		Vec2(T xValue, T yValue) : x(xValue), y(yValue) {}
	
		//casting operator
		template<typename A> operator Vec2<A>() const {
			return Vec2<A>(static_cast<A>(x), static_cast<A>(y));
		}
		//equality operator
		template<typename A> bool operator==(const Vec2<A> o) const {
			return (x == o.x && y == o.y);
		}
		//inequality operator
		template<typename A> bool operator!=(const Vec2<A> o) const {
			return (x != o.x || y != o.y);
		}

		//addition operator
		template<typename A> Vec2<T> operator+(const Vec2<A> o) const {
			return Vec2<T>(x + o.x, y + o.y);
		}
		//addition operator
		template<typename A> Vec2<T>& operator+=(const Vec2<A> o) {
			x += static_cast<T>(o.x);
			y += static_cast<T>(o.y);
			return *this;
		}

		//subtraction operator
		template<typename A> Vec2<T> operator-(const Vec2<A> o) const {
			return Vec2<T>(x - o.x, y - o.y);
		}
		//subtraction operator
		template<typename A> Vec2<T>& operator-=(const Vec2<A> o) {
			x -= static_cast<T>(o.x);
			y -= static_cast<T>(o.y);
			return *this;
		}

		//multiplication operator
		template<typename A> Vec2<T> operator*(const Vec2<A> o) const {
			return Vec2<T>(x * o.x, y * o.y);
		}
		//multiplication operator
		template<typename A> Vec2<T>& operator*=(const Vec2<A> o) {
			x *= static_cast<T>(o.x);
			y *= static_cast<T>(o.y);
			return *this;
		}

		//division operator
		template<typename A> Vec2<T> operator/(const Vec2<A> o) const {
			return Vec2<T>(x / o.x, y / o.y);
		}
		//division operator
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