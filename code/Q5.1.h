/*
You are given two 32-bit numbers, N and M, and two bit positions, i and j. Write a method to set all bits between i and j in N equal to M (e.g., M becomes a substring of N located at i and starting at j).

EXAMPLE:

Input: N = 10000000000, M = 10101, i = 2, j = 6

Output: N = 10001010100
*/

int update_bits(int n, int m, int i, int j){
    int max = ~0;   //all 1s
    int left = max<<(j+1);
    int right = ((1 << i) -1);
    int mask = left | right;
    return (n & mask) | (m << i);
}

/*
C++中关于位操作，记录几点需要注意的地方：

一个有符号数，如果它的最高位为1，它右移若干位后到达位置i， 那么最高位到第i位之间全是1，例如：
int a = 1;
a <<= 31;    //a:1后面带31个0
a >>= 31;    //a:32个1，即-1
cout<<a<<endl;    //输出-1(写下负号，然后取反加1)

一个无符号数，如果它的最高位为1，它右移若干位后到达位置i， 那么最高位到第i位之间全是0，例如：
unsigned int a = 1;
a <<= 31;    //a:1后面带31个0
a >>= 31;    //a:31个0后面带一个1，即1
cout<<a<<endl;    //输出1
*/