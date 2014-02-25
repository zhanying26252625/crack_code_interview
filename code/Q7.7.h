/*
Design an algorithm to find the kth number such that the only prime factors are 3, 5, and 7.
*/

// Naive way 一种简单的思路就是对于已经列出的数，我们依次去乘以3，5，7得到一组数 然后找出最小且还没有列出的数，加入到这个列表。然后重复上面的步骤： 
// 乘以3，5，7，找出最小且还没有列出的数……这个方法的时间复杂度是O(n2 )。
// O(N^2)

#include <iostream>
#include <queue>
using namespace std;

int mini(int a, int b){
    return a < b ? a : b;
}
int mini(int a, int b, int c){
    return mini(mini(a, b), c);
}

//O(N)
int get_num(int k){
    if(k <= 0) return 0;
    int res = 1, cnt = 1;
    queue<int> q3, q5, q7;
    q3.push(3); q5.push(5); q7.push(7);
    for(; cnt<k; ++cnt){
        int v3 = q3.front();
        int v5 = q5.front();
        int v7 = q7.front();
		/*
		for lots of primes we can also use priority_queue<ListNode*, deque<ListNode*>, MyComp > 
		and unordered_map to solve it effectively
		*/
        res = mini(v3, v5, v7); 
		if(res == v7){
            q7.pop();
        }
        else{
            if(res == v5){
                q5.pop();
            }
            else{
                if(res == v3){
                    q3.pop();
                    q3.push(3*res);
                }
            }
            q5.push(5*res);
        }
        q7.push(7*res);
    }
    return res;
}
int main(){
    for(int i=1; i<20; ++i)
        cout<<get_num(i)<<endl;
    return 0;
}