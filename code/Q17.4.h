/*
Write a method which finds the maximum of two numbers. You should not use if-else or any other comparison operator.

EXAMPLE

Input: 5, 10

Output: 10
*/

#include <iostream>
using namespace std;

//assume no overflown, otherwise we can use long long , or use string for arbitrary large of number

int Max1(int a, int b){
    int c[2] = {
        a, b
    };
    int z = a - b;
    z = (z>>31) & 1;
    return c[z];
}

int Max2(int a, int b){
    int z = a - b;
    int k = (z>>31) & 1;
    return a - k * z;
}





