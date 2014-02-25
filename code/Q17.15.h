/*
Max subarray sum
Max subarray product
*/

//sum O(N)
int maxSubArray(const vector<int>& v){
    int max = 0;
    int curMax = 0;
    for(int i = 0; i < v.size(); ++i){
        if(curMax<0) curMax=0;
        curMax += v[i];
        max = max>curMax?max:curMax;
    }
    return max;
}

//product O(N^2)
/*
 * 求解一个数组相邻数相乘的最大值 例如 1 2 3 0 3 7 :21 方法一：时间复杂度为O(n*n),统计每一个子序列乘积
 */
 
     int computeMultip1(const vector<int>& numb) {
         int result = INT_MIN;
         for (int i = 0; i < numb.size(); i++) {
			int tmpresult = 1;
             for (int j = i; j < numb.size(); j++) {
                 tmpresult *= numb[j];
                 if (tmpresult > result)
                     result = tmpresult;
             }
         }
         return result;
     }
	 
//DP O(N) ************
 	  /*
      通过动态规划求解，考虑到可能存在负数的情况，
	  我们用Max[i],来表示以a[i]结尾的最大连续子序列的乘积，
	      用Min[i]表示以a[i]结尾的最小的连续子序列的乘积值 
	  那么状态转移方程为： Max[i]=max{a[i],Max[i-1]*a[i], Min[i-1]*a[i]}; 
						   Min[i]=min{a[i], Max[i-1]*a[i], Min[i-1]*a[i]}; 
	  初始状态为Max[1]=Min[1]=a[1]     

      */
	 int computeMultip2(const vector<int>& numb) {
		 vector<int> maxMul(numb.size(),1);
		 vector<int> minMul(numb.size(),1);
         maxMul[0] = numb[0];
         minMul[0] = numb[0];
         int maxValue = numb[0];
         for (int i = 1; i < minMul.size(); ++i) {
             maxMul[i] = std::max(std::max(numb[i], maxMul[i - 1] * numb[i]),
                                  minMul[i - 1] * numb[i]);
             minMul[i] = std::min(std::min(numb[i], maxMul[i - 1] * numb[i]),
                                  minMul[i - 1] * numb[i]);
             maxValue = std::max(maxMul[i], maxValue);
         }
         return maxValue;
     }
int main(){
	int array[]={1, -2, -3, 0, 7, -8, -2 };
	cout<<computeMultip2( vector<int>(array,array+sizeof(array)/sizeof(array[0])) );
	getchar();
	return 1;
}