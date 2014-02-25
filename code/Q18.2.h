/*
Write a method to shuffle a deck of cards. It must be a perfect shuffle - 
in other words, each 52! permutations of the deck has to be equally likely.
Assume that you are given a random number generator which is perfect
*/

/*
最直观的思路是什么？很简单，每次从牌堆中随机地拿一张出来。那么， 第一次拿有52种可能，拿完后剩下51张；
第二次拿有51种可能，第三次拿有50种可能， …，
一直这样随机地拿下去直到拿完最后1张，我们就从52!种可能中取出了一种排列， 这个排列对应的概率是1/(52!)，
正好是题目所要求的。
*/

#include <iostream>
#include <cstdlib>
using namespace std;

void Swap(int &a, int &b){// 有可能swap同一变量，不能用异或版本!!!!!!!!!
    int t = a;
    a = b;
    b = t;
}

//O(N)
void RandomShuffle(int a[], int n){
    for(int i=0; i<n; ++i){
        int j = rand() % (n-i) + i;// 产生 i 到 n-1 间的随机数
        Swap(a[i], a[j]);
    }
}
int main(){
    srand((unsigned)time(0));
    int n = 9;
    int a[] = {
        1, 2, 3, 4, 5, 6, 7, 8, 9
    };
    RandomShuffle(a, n);
    for(int i=0; i<n; ++i)
        cout<<a[i]<<endl;
    return 0;
}

/*
similar problem
Write a method to randomly generate a set of m integers from an array of size n. 
Each element must have equal probability of being chosen.

选第1个元素：在n个中随机选，因此概率为1/n

选第2个元素：在剩下的n-1个中随机选：1/(n-1)，由于第1次没有选中它， 而是在另外n-1个中选：(n-1)/n，因此概率为：(n-1)/n * 1/(n-1) = 1/n

选第3个元素：同上：(n-1)/n * (n-2)/(n-1) * 1/(n-2) = 1/n
*/
void PickMRandomly(int a[], int n, int m){
    for(int i=0; i<m; ++i){
        int j = rand() % (n-i) + i;// 产生i到n-1间的随机数
        Swap(a[i], a[j]);
    }
}

