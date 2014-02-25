/*
Imagine you have a square matrix, where each cell is filled with either black or white. 
Design an algorithm to find the maximum subsquare such that all four borders are filled with black pixels.
*/

/*
Naive O(N^4)
*/
bool IsSquare(int row, int col, int size){
    for(int i=0; i<size; ++i){
        if(matrix[row][col+i] == 1)// 1代表白色，0代表黑色
            return false;
        if(matrix[row+size-1][col+i] == 1)
            return false;
        if(matrix[row+i][col] == 1)
            return false;
        if(matrix[row+i][col+size-1] == 1)
            return false;
    }
    return true;
}

SubSquare FindSubSquare(int n){
    int max_size = 0; //最大边长
    int col = 0;
    SubSquare sq;
    while(n-col > max_size){
        for(int row=0; row<n; ++row){
            int size = n - max(row, col);
            while(size > max_size){
                if(IsSquare(row, col, size)){
                    max_size = size;
                    sq.row = row;
                    sq.col = col;
                    sq.size = size;
                    break;
                }
                --size;
            }
        }
        ++col;
    }
    return sq;
}

/*
O(N^3) clever, pre-processing

1. construct a row and a col matrix to record the # of consecutive "1" from current node to left, and up, respectively. 

e.g.: 

Original (A): 
11111 
11101 
10111 
01011 

row: 
12345 
12301 
10123 
01012 

col: 
11111 
22202 
30313 
01024
*/

int maxsize = 0;

for i=0; i<N; i++
for j=0; j<N, j++{
    if (A[i,j]==1){
	for (int k=min(i,j); k>maxsize; k--){ //min(i,j) is the maximum possible square size from i,j to 0,0 
	   if ((row[i,j] - row[i,j-k] == k) && 
	   (row[i-k,j] - row[i-k,j-k] == k) && 
	   (col[i,j] - col[i-k,j] == k) &&	
	   (col[i-k,j] - col[i-k,j-k] == k) &&
	   )
	   maxsize=k;
	   break;
	}
    }
}
