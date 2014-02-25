/*
Given an array of integers, find the minimal indexes range n and m such that once 
this is range is in order, the all array would be in order

input : 1,2,4,7,10,11,7,12,6,7,16,18,19
output: 3,9

4 algorithms

*/

/*
O(N^2) Naive
pass 1, from left to right, test whether each is less than all to its right
pass 2, from right to left, test whether each is larger than all to its left
*/

/*
O(NlgN) stable sorting with original index booked
pair<int,int>(); value,origin_index
after sorting, find the range that index!=origin_index
*/

/*
O(N) CTCI algo,
find three region, first and last are in order.
expand middle unordered range to left and right as much as possbile
1,2,4,7,10,11,  7,12,  6,7,16,18,19
*/


void findRange(int array[] ,int size, int& n, int& m){
	n=0;
	m=0;
	//find left region in order
	int i = 1;
	for(; i < size; ++i){
		if(array[i]<array[i-1])break;
	}
	int leftEnd=i-1;
	if(leftEnd==size-1) return; // all are in order

	//find right region in order
	i = size-2;
	for(; i >= 0; --i){
		if(array[i]>array[i+1])break;
	}
	int rightBegin=i+1;
	if(rightBegin==0) return; // all are in order

	//find min,max in middle unordered range 
	i = leftEnd+1;
	int min=INT_MAX;
	int max=INT_MIN;
	for(;i < rightBegin ; ++i){
		min = std::min(min,array[i]);
		max = std::max(max,array[i]);
	}

	//expand middle range as far as possible
	while(leftEnd>=0&&array[leftEnd]>min) --leftEnd;
	while(rightBegin<size&&array[rightBegin]<max) ++rightBegin;

	n=leftEnd+1;
	m=rightBegin-1;
}
/*
unfortunately, this is a wrong algo. won't work for  1,2,4,7,10,11,  7,12,  1,7,16,18,19
to fix the problem we should partitiaon like this:
1,2,4,7,10,11,      7,12,1,7,16,18,19, minRight=1
1,2,4,7,10,11,7,12,     1,7,16,18,19, minLeft=12

see my algo below
*/

/*
O(N) my algo
*/
void findRange1(int array[] ,int size, int& n, int& m){
	n=0;
	m=0;
	//find left region in order
	int i = 1;
	for(; i < size; ++i) if(array[i]<array[i-1])break;
	int leftEnd=i-1;
	if(leftEnd==size-1) return; // all are in order

	//find right region in order
	i = size-2;
	for(; i >= 0; --i) if(array[i]>array[i+1])break;
	int rightBegin=i+1;
	if(rightBegin==0) return; // all are in order

	//find min,max 
	int min=INT_MAX;
	int max=INT_MIN;
	for(i = leftEnd+1; i < size ; ++i) min = std::min(min,array[i]);
	for(i = rightBegin-1; i >= 0; --i) max = std::max(max,array[i]);	

	//shrink both region
	while(leftEnd>=0&&array[leftEnd]>min) --leftEnd;
	while(rightBegin<size&&array[rightBegin]<max) ++rightBegin;

	n=leftEnd+1;
	m=rightBegin-1;
}




















