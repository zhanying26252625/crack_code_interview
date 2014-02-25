/*
Given an NxN matrix of positive and negative integers, write code to find the submatrix with the largest possible sum.

*/

/*
暴力法，时间复杂度O(n6 )

最简单粗暴的方法就是枚举所有的子矩阵，求和，然后找出最大值。 
枚举子矩阵一共有C(n, 2)*C(n, 2)个(水平方向选两条边，垂直方向选两条边)， 
时间复杂度O(n4 )，求子矩阵中元素的和需要O(n2 )的时间。 因此总的时间复杂度为O(n6 )。
*/

/*
O(N^4) DP
*/
#include <iostream>
#include <climits>
#define N 3
using namespace std;
int sumMatrix[N][N];
void preComputeMatrix(int a[][N])
{
    for(int i=0; i<N; i++)
    {
        for(int j=0; j<N; j++)
        {
            if(i==0 && j==0)
                sumMatrix[i][j] = a[i][j];
            else if(i==0)
                sumMatrix[i][j]+=sumMatrix[i][j-1] + a[i][j];
            else if(j==0)
                sumMatrix[i][j]+=sumMatrix[i-1][j] + a[i][j];
            else
                sumMatrix[i][j]+=sumMatrix[i-1][j]+sumMatrix[i][j-1]-sumMatrix[i-1][j-1] + a[i][j];
        }
    }
}
int computeSum(int a[][N],int i1,int i2,int j1,int j2)
{
    if(i1==0 && j1==0)
        return sumMatrix[i2][j2];
    else if(i1==0)
        return sumMatrix[i2][j2] - sumMatrix[i2][j1-1];
    else if(j1==0)
        return sumMatrix[i2][j2] - sumMatrix[i1-1][j2];
    else
        return sumMatrix[i2][j2] - sumMatrix[i2][j1-1]- sumMatrix[i1-1][j2] + sumMatrix[i1-1][j1-1];
}
int getMaxMatrix(int a[][N])
{
    int maxSum = INT_MIN;
    for(int row1=0; row1<N; row1++)
    {
        for(int row2=row1; row2<N; row2++)
        {
            for(int col1=0; col1<N; col1++)
            {
                for(int col2=col1; col2<N; col2++)
                {
                    maxSum = max(maxSum,computeSum(a,row1,row2,col1,col2));
                }
            }
        }
    }
    return maxSum;
}
int main( void )
{
    int a[N][N] = {{-1,-2,-3},{-10,-5,-15},{6,-8,-20}};
    preComputeMatrix(a);
    for(int i=0; i<N; i++)
    {
        for(int j=0; j<N; j++)
            cout<<sumMatrix[i][j]<<" ";
        cout<<endl;
    }
    cout<<getMaxMatrix(a);
    return 0;
}



/*
O(N^3) BEST
*/
int findMaxSum (int matrix[numRows][numCols])
{
    int maxSum=0;
 
    for (int left = 0; left < numCols; left++)
    {
        int temp[numRows] = {0};
 
        for (int right = left; right < numCols; right++)
        {
            // Find sum of every mini-row between left and right columns and save it into temp[]
            for (int i = 0; i < numRows; ++i)
                temp[i] += matrix[i][right];
 
            // Find the maximum sum subarray in temp[].
            int sum = kadane(temp, numRows);
 
            if (sum > maxSum)
                maxSum = sum;
        }
    }
 
    return maxSum;
}

//kadane
int maxSubArray(const vector<int>& v){
    int max = 0;
    int curMax = 0;

    for(int i = 0; i < v.size(); ++i){
        if(curMax<0)
            curMax=0;
        
        curMax += v[i];
        max = max>curMax?max:curMax;
    }

    return max;
}

int main( void )
{
    int n;
    cin>>n;
    for(int i=0;i<n;i++)
    {
        for(int j=0;j<n;j++)
        {
            cin>>a[i][j];
        }
    }
    cout<<kadane2D(n)<<endl;
    return 0;
}