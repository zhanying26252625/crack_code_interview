/*
Given an integer, print the next larger number that have the same number of 1 bits in their binary representation.

Suppose we have a pattern of N bits set to 1 in an integer and we want the next permutation of N 1 bits in a lexicographical sense.
 For example, if N is 3 and the bit pattern is 00010011, the next patterns would be 00010101, 00010110, 00011001,00011010, 00011100, 00100011, and so forth.
 The following is a fast way to compute the next permutation.
 
 unsigned int t = (v | (v - 1)) + 1;  
w = t | ((((t & -t) / (v & -v)) >> 1) - 1);
*/

// this function returns next higher number with same number of set bits as x.
/*
011 -> 101
011000 -> 100001
*/
uint_t snoob(uint_t x)
{
  uint_t rightOne;
  uint_t nextHigherOneBit;
  uint_t rightOnesPattern;
  uint_t next = 0;
 
 // x = aaa0 1111 0000 - > aaa1 0000 0111
  if(x) //aaa0 1111 0000
  {
    // right most set bit
    rightOne = x & -(signed)x; // 0000 0001 0000
	
    // reset the pattern and set next higher bit
    nextHigherOneBit = x + rightOne; //aaa1 0000 0000
	
    // isolate the pattern
    rightOnesPattern = x ^ nextHigherOneBit; //0001 1111 0000
	
    // right adjust pattern
    rightOnesPattern = (rightOnesPattern>>2)/rightOne; // 0000 0111 1100  /  0000 0001 0000  = 0000 0000 0111

    next = nextHigherOneBit | rightOnesPattern; // aaa1 0000 0111
  }
 
  return next;
}