/*
Write a program to 
swap odd and even bits in an integer with as few instructions as possible 
(e.g., bit 0 and bit 1 are swapped, bit 2 and bit 3 are swapped, etc).
*/

int swap_bits(int x){
    return ((x & 0x55555555) << 1) | ((x >> 1) & 0x55555555);
}