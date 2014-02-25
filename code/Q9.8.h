/*
Given an infinite number of quarters 
(25 cents), dimes (10 cents), nickels (5 cents) and pennies (1 cent),
 write code to calculate the number of ways of representing n cents.
*/

//wrong answer
int cnt = 0;
void sumn(int sum, int n){
    if(sum >= n){
        if(sum == n) ++cnt;
        return;
    }
    else{
        sumn(sum+25, n);
        sumn(sum+10, n);
        sumn(sum+5, n);
        sumn(sum+1, n);
    }
}
//问题出在哪？有序与无序的区别！这个函数计算出来的组合是有序的， 
//也就是它会认为1,5和5,1是不一样的，导致计算出的组合里有大量是重复的

//corrent 1
int make_change(int n, int denom){
    int next_denom = 0;
    switch(denom){
    case 25:
        next_denom = 10;
        break;
    case 10:
        next_denom = 5;
        break;
    case 5:
        next_denom = 1;
        break;
    case 1:
        return 1;
    }
    int ways = 0;
    for(int i=0; i*denom<=n; ++i)
        ways += make_change(n-i*denom, next_denom);
    return ways;
}

//corret 2 better!
for denoms
	for i from 1 to n
		dp[i] += dp[i-denom];
return dp[n]


