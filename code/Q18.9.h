/*
online median
Numbers are randomly generated and passed to a method. 
Write a program to find and maintain the median value as new values are generated.
*/

#include<set>
#include<queue>
#include <limits>
#include <cassert>
#include <cmath>
#include <sstream>
//#include <unordered_map>

using namespace std;

void online_median(istringstream* sin) {
  // Min-heap stores the bigger part of the stream.
  priority_queue<int, vector<int>, greater<int> > H;
  // Max-heap stores the smaller part of the stream.
  priority_queue<int, vector<int>, less<int> > L;

  int x;
  while (*sin >> x) {

    if (!L.empty() && x > L.top()) {
      H.push(x);
    } else {
      L.push(x);
    }

    //balance
    if (H.size() > L.size() + 1) {
      L.push(H.top());
      H.pop();
    } else if (L.size() > H.size() + 1) {
      H.push(L.top());
      L.pop();
    }

    if (H.size() == L.size()) {
      cout << 0.5 * (H.top() + L.top()) << endl;
    } else {
      cout << (H.size() > L.size() ? H.top() : L.top()) << endl;
    }
  }
}