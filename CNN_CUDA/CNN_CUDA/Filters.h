#pragma once
#include <vector>

using namespace std;

class Filters
{
public:

	const int size = 3;
	const int num = 9;

	const vector<vector<int>> verticalLine{
		{0, 1, 0},
		{0, 1, 0},
		{0, 1, 0},
	};
	int verticalSum = 0;

	const vector<vector<int>> horizontalLine{
		{0, 0, 0},
		{1, 1, 1},
		{0, 0, 0},
	};
	int horizontalSum = 0;

	const vector<vector<int>> leftDiagonalLine{
		{0, 0, 1},
		{0, 1, 0},
		{1, 0, 0},
	};
	int leftDiagonalSum = 0;

	const vector<vector<int>> rightDiagonalLine{
		{1, 0, 0},
		{0, 1, 0},
		{0, 0, 1},
	};
	int rightDiagonalSum = 0;

	const vector<vector<int>> cross{
		{0, 1, 0},
		{1, 1, 1},
		{0, 1, 0},
	};
	int crossSum = 0;

	const vector<vector<int>> plus{
		{1, 1, 1},
		{1, 1, 1},
		{1, 1, 1},
	};
	int plusSum = 0;

	const vector<vector<int>> x{
		{1, 0, 1},
		{0, 1, 0},
		{1, 0, 1},
	};
	int xSum = 0;

	const vector<vector<int>> square{
		{1, 1, 1},
		{1, 0, 1},
		{1, 1, 1},
	};
	int squareSum = 0;

	const vector<vector<int>> diamond{
		{0, 1, 0},
		{1, 0, 1},
		{0, 1, 0},
	};
	int diamondSum = 0;


	bool cleanSum() {
		this->verticalSum = 0;
		this->horizontalSum = 0;
		this->leftDiagonalSum = 0;
		this->rightDiagonalSum = 0;
		this->crossSum = 0;
		this->plusSum = 0;
		this->xSum = 0;
		this->squareSum = 0;
		this->diamondSum = 0;
		return true;
	}
};