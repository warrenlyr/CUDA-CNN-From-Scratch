#include <vector>

using namespace std;

class Filter
{
public:

	const int size = 3;
	const int num = 9;

	const vector<vector<int>> verticalLine{
		{0, 1, 0},
		{0, 1, 0},
		{0, 1, 0},
	};

	const vector<vector<int>> horizontalLine{
		{0, 0, 0},
		{1, 1, 1},
		{0, 0, 0},
	};

	const vector<vector<int>> leftDiagonalLine{
		{0, 0, 1},
		{0, 1, 0},
		{1, 0, 0},
	};

	const vector<vector<int>> rightDiagonalLine{
		{1, 0, 0},
		{0, 1, 0},
		{0, 0, 1},
	};

	const vector<vector<int>> cross{
		{0, 1, 0},
		{1, 1, 1},
		{0, 1, 0},
	};

	const vector<vector<int>> plus{
		{1, 1, 1},
		{1, 1, 1},
		{1, 1, 1},
	};

	const vector<vector<int>> x{
		{1, 0, 1},
		{0, 1, 0},
		{1, 0, 1},
	};

	const vector<vector<int>> square{
		{1, 1, 1},
		{1, 0, 1},
		{1, 1, 1},
	};

	const vector<vector<int>> diamond{
		{0, 1, 0},
		{1, 0, 1},
		{0, 1, 0},
	};
};