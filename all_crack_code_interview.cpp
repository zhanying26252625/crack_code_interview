//Q2.5.h


/*
Given a circular linked list, implement an algorithm which returns node at the beginning of the loop.

DEFINITION

Circular linked list: A (corrupt) linked list in which a node’s next pointer points to an earlier node, so as to make a loop in the linked list.

EXAMPLE

Input: A -> B -> C -> D -> E -> C [the same C as earlier]

Output: C

译文：

给定一个循环链表，实现一个算法返回这个环的开始结点。

定义：

循环链表：链表中一个结点的指针指向先前已经出现的结点，导致链表中出现环。

例子：

输入：A -> B -> C -> D -> E -> C [结点C在之前已经出现过]

输出：结点C
*/

node* loopstart(node *head){
    if(head==NULL) return NULL;
    node *fast = head, *slow = head;
    while(fast && fast->next){
        fast = fast->next->next;
        slow = slow->next;
        if(fast==slow) break;
    }
    if(!fast || !fast->next) return NULL;
    slow = head;
    while(fast!=slow){
        fast = fast->next;
        slow = slow->next;
}
    
/*
Let us denote the length of non-loop list as m, 
Let's also have the size of the loop as L. 
When the two pointer meets, let us denote the distance between the meeting point of the starting of the loop as d.

So, at the time of their meet, the fast pointer would have travelled m+nL-d  
and the slow pointer only travelled m+L-d . 
Therefore we have m+nL-d=2(m+L-d)
=> d m (mod L)

Therefore, when the two pointer meets, we can move one pointer to the head,
and then move them one step at each time, when they meet again, they will meet at the starting pointer of the loop
*/
//Q2.7.h


/*
Check whether a list is a palindrome

1	reverse				
2	to array				
3	stack first half and the compare with remaining half				

*/
//Q3.6.h


/*
Write a program to sort a stack in ascending order.
 You should not make any assumptions about how the stack is implemented. 
 The following are the only functions that should be used to write this program: push | pop | peek | isEmpty.
*/

//iterative O(N^2)
stack<int> Ssort(stack<int> s){
    stack<int> t;
    while(!s.empty()){
        int data = s.top();
        s.pop();
        while(!t.empty() && t.top()>data){
            s.push(t.top());
            t.pop();
        }
        t.push(data);
    }
    return t;
}

//recursive O(N^2)
void Ssort(stack<int>& s){
	if(s.empty())
		return;
	int data = s.top();
	s.pop();	
	Ssort(s);
	insert(s,data);
}

void insert(stack<int>& s, int data){
	if(s.empty()||s.top()<data){
		s.push(data);
	}
	else{
		int d = s.top();
		s.pop();
		insert(s,data);
		s.push(d);
	}
}
//Q4.6.h


/*
find next node in BST, if has parent link, what about there is no parent link
*/

struct node * minValue(struct node* node) {
  struct node* current = node;
  
  /* loop down to find the leftmost leaf */
  while (current->left != NULL) {
    current = current->left;
  }
  return current;
}


//parent link
struct node * inOrderSuccessor(struct node *root, struct node *n)
{
  // step 1 of the above algorithm 
  if( n->right != NULL )
    return minValue(n->right);
 
  // step 2 of the above algorithm
  struct node *p = n->parent;
  while(p != NULL && n == p->right)
  {
     n = p;
     p = p->parent;
  }
  return p;
}

//no parent link
struct node * inOrderSuccessor(struct node *root, struct node *n)
{
    // step 1 of the above algorithm
    if( n->right != NULL )
        return minValue(n->right);
 
    struct node *succ = NULL;
 
    // Start from root and search for successor down the tree
    while (root != NULL)
    {
        if (n->data < root->data)
        {
            succ = root;
            root = root->left;
        }
        else if (n->data > root->data)
            root = root->right;
        else
           break;
    }
 
    return succ;
}
//Q4.7.h


/*
first common ancestor of two nodes in binary tree, how about it's BST
*/

//general O(N)
Node *LCA(Node *root, Node *p, Node *q) {
  if (!root) return NULL;
  if (root == p || root == q) return root;
  Node *L = LCA(root->left, p, q);
  Node *R = LCA(root->right, p, q);
  if (L && R) return root;  // if p and q are on both sides
  return L ? L : R;  // either one of p,q is on one side OR p,q is not in L&R subtrees
}

//BST O(lgN)
Node *LCA(Node *root, Node *p, Node *q) {
  if (!root || !p || !q) return NULL;
  if (max(p->data, q->data) < root->data)
    return LCA(root->left, p, q);
  else if (min(p->data, q->data) > root->data)
    return LCA(root->right, p, q);
  else
    return root;
}

//Offline O(1)
/*
Tarjan's off-line lowest common ancestors algorithm, RMQ
http://en.wikipedia.org/wiki/Tarjan's_off-line_lowest_common_ancestors_algorithm

http://www.topcoder.com/tc?d1=tutorials&d2=lowestCommonAncestor&module=Static

*/
//Q4.8.h


/*
You are given a binary tree in which each node contains a value. 
Design an algorithm to print all paths which sum up to that value. 
Note that it can be any path in the tree - it does not have to start at the root.
*/

void find_sum2(Node* head, int sum, vector<int>& v){
    if(head == NULL) return;
    v.push_back(head->key);
    int tmp = 0;
    for(int i=v.size()-1; i>=0; --i){
        tmp += v[i];
        if(tmp == sum)
            print2(v, i);
    }
    find_sum2(head->lchild, sum, v);
    find_sum2(head->rchild, sum, v);
	v.pop_back()
}

void print2(vector<int> v, int level){
    for(int i=level; i<v.size(); ++i)
        cout<<v.at(i)<<" ";
    cout<<endl;
}
//Q5.1.h


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
//Q5.3.h


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
//Q5.6.h


/*
Write a program to 
swap odd and even bits in an integer with as few instructions as possible 
(e.g., bit 0 and bit 1 are swapped, bit 2 and bit 3 are swapped, etc).
*/

int swap_bits(int x){
    return ((x & 0x55555555) << 1) | ((x >> 1) & 0x55555555);
}
//Q7.7.h


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
//Q9.4.h


/*
Write a method that returns all subsets of a set.
*/

//如果一个集合有n个元素，那么它可以用0到2n -1 总共2n 个数的二进制形式来指示
typedef vector<vector<int> > vvi;
typedef vector<int> vi;
vvi get_subsets(int a[], int n){ //O(n2^n)
    vvi subsets;
    int max = 1<<n;
    for(int i=0; i<max; ++i){
        vi subset;
        int idx = 0;
        int j = i;
        while(j > 0){
            if(j&1){
                subset.push_back(a[idx]);
            }
            j >>= 1;
            ++idx;
        }
        subsets.push_back(subset);
    }
    return subsets;
}

//leetcode subset
class Solution {
public:

	void subsets(vector<int> &S, int index, vector<int>& v, vector<vector<int> >& vv) {

		vv.push_back(v);

		if(index<S.size()){
			for(int i = index; i < S.size(); ++i){
				v.push_back(S[i]);
				subsets(S,i+1,v,vv);
				v.pop_back();
			}
		}
	}

    vector<vector<int> > subsets(vector<int> &S) {
        vector<vector<int> > vv;
        vector<int> v;
        sort(S.begin(),S.end());
        subsets(S,0,v,vv);
        return vv;
    }

};

//based on bits operation
void generatePowerset1(const vector<int>& v){
   int num = 0;
   while( num < 1<<v.size() ){
        int n(num) ;
        while(n){
            int index = (int)log2( n & (~(n-1)) );
            cout << v[index]<<" ";
            n &= (n-1);
        }
        cout<<endl;
        ++num;
   }
}

void generatePowerset2Helper(const vector<int>& v,int start,vector<int>* retV){
    //print
    copy(retV->begin(),retV->end(),ostream_iterator<int>(cout," "));
    cout<<endl;
   
    for(int i=start; i < v.size(); ++i){
        retV->push_back(v[i]);
        generatePowerset2Helper(v,i+1,retV);
        retV->pop_back();
    }
}
    
//based on recursion
void generatePowerset2(const vector<int>& v){
    vector<int> retV;
    generatePowerset2Helper(v,0,&retV);
}
//Q9.5.h


/*
Write a method to compute all permutations of a string
*/

/*
我们可以把串“abc”中的第0个字符a取出来，然后递归调用permu计算剩余的串“bc” 的排列，得到{bc, cb}。
然后再将字符a插入这两个串中的任何一个空位(插空法)， 得到最终所有的排列。比如，a插入串bc的所有(3个)空位，得到{abc,bac,bca}。 
递归的终止条件是什么呢？当一个串为空，就无法再取出其中的第0个字符了， 所以此时返回一个空的排列
*/
typedef vector<string> vs;

vs permu(string s){
    vs result;
    if(s == ""){
        result.push_back("");
        return result;
    }
    string c = s.substr(0, 1);
    vs res = permu(s.substr(1));
    for(int i=0; i<res.size(); ++i){
        string t = res[i];
        for(int j=0; j<=t.length(); ++j){
            string u = t;
            u.insert(j, c);
            result.push_back(u);
        }
    }
    return result; 
}

//leetcode
/*
Given a collection of numbers, return all possible permutations.

For example,
[1,2,3] have the following permutations:
[1,2,3], [1,3,2], [2,1,3], [2,3,1], [3,1,2], and [3,2,1].
*/

class Solution {
public:

    vector<vector<int> > retVV;
    
    void permuteHepler(vector<int> &num ,vector<bool>& used, vector<int>& v) {
        if(v.size() == num.size()){
            retVV.push_back(v);
            return;
        }
        
        for(int i = 0; i < num.size(); ++i){
            if( ! used[i] ){
                used[i]=true;
                v.push_back(num[i]);
                permuteHepler(num,used,v);
                used[i]=false;
                v.pop_back();
            }
        }
    }
    
    vector<vector<int> > permute(vector<int> &num) {
        vector<int> v;
        vector<bool> used(num.size(),false);
        permuteHepler(num,used,v);
        return retVV;
    }
    
};
//Q9.8.h


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



//Q9.10.h


/*
build height sequence for N boxes each with distinct height,width and depth
*/

//similar to longest subsequence of an array, here each element is box not integer


//DP O(N^2)
vector<int> longestSubSequence(const vector<int>& v){

    vector<int> ret;

    //pair < len, previsou index >
    vector< pair<int,int> > records(v.size(), pair<int,int>(1,-1));

    int maxLen = 0;
    int maxIndex = 0;
    for(int i = 1; i< v.size(); ++i){
        for(int j =0 ;j < i; ++j){
            if( v[i] > v[j] && records[i].first < records[j].first+1 ){
                records[i].first = records[j].first+1;
                records[i].second = j;
            }
        }
        if(records[i].first > maxLen){
            maxLen = records[i].first;
            maxIndex = i;
        }
    }

    //re-construct the longest sequence, decreasing order
    int pre = maxIndex; 
    while(pre>=0){
        ret.push_back(v[pre]);
        pre = records[pre].second;
    }

    //increasing order
    reverse(ret.begin(),ret.end());

    return ret;
}

//DP O(NlgN), advanced and clever
//Using binary search to find, instead of previous solution which iterator through 0->i
//So clever a solution
vector<int> longestSubSequence2(const vector<int>& v){

    vector<int> ret;

    for(int i = 0; i < v.size(); ++i){
        //binary search
        vector<int>::iterator iter = lower_bound(ret.begin(),ret.end(),v[i]);        
        if(iter==ret.end()){
            ret.push_back(v[i]);
        }
        else{
            *iter = v[i];
        }
    }

    return ret;
}
//Q9.11.h


/*
find number of ways to parenthesize a boolean expression(&,^,|) to achieve a given result
*/

/*
algo, breakdown each boolean operation

f(e1|e2,true) = f(e1,true) * f(e2,true) + f(e1,false) * f(e2,true) + f(e1,true) * f(e2,false)
etc

cache intermediate result.
key could be expr+":"+result for f(expr,result)
*/
//Q10.8.h


/*
a series of number is coming in a streaming fashion, implement a data structure that would tell u the rank of each number effectively

*/

//solution
/*
A modified BST with additional property(number of nodes on left-side) would serve the purpose.
Effective for both insertion and getRank O(lgN)

1.hashtable if not good since no ordering info
2.heap is not good since only the top is known
3.array is not good for insertion
4.list is not good for getRank
*/

/*
			20(4)
		15(3)     25(2)
	10(1)		23(0)
5(0)    13(0)		24(0)
*/	

int getRank(Node* root, int x){
	if(root){
		if(root->val==x)
			return root->rank;
		else if(root->val>x)
			return getRank(root->left,x);
		else if(root->right)
			return root->rank+1+getRank(root->right,x);
		else
			return -1;
	}
	return -1;
}
//Q12.3.h


/*
Given an input file with four billion integers, provide an algorithm to generate an integer which is not contained in the file. Assume you have 1 GB of memory.

FOLLOW UP

What if you have only 10 MB of memory?
*/

//1G, bit map
int main(){

    int int_len = sizeof(int) * 8;
    int bit_len = 0xFFFFFFFF / int_len;
    int* bit = new int[bit_len];
    int v;
    while(scanf("%d", &v) != EOF){
        bit[v/int_len] |= 1<<(v%int_len);
    }
    bool found = false;
    for(int i=0; i<bit_len; ++i){
        for(int j=0; j<int_len; ++j){
            if((bit[i] & (1<<j)) == 0){
                cout<<i*int_len + j<<endl;
                found = true;
                break;
            }
                
        }
        if(found) break;
    }
    
    delete[] bit;
    fclose(stdin);
    return 0;
}

//10MG, two pass
int main(){
    freopen("12.3.in", "r", stdin);// 20000 number
    int int_len = sizeof(int) * 8;
    int totalnum = 20000;
    int blocksize = 2000;
    int blocknum = totalnum / blocksize;
    int* block = new int[blocknum];
    int* bit = new int[blocksize/int_len+1];
    int v;
    while(scanf("%d", &v) != EOF){
        ++block[v/blocksize];
    }
    fclose(stdin);
    int start;
    for(int i=0; i<blocknum; ++i){
        if(block[i] < blocksize){
            start = i * blocksize;
            break;
        }
    }
    freopen("12.3.in", "r", stdin);
    while(scanf("%d", &v) != EOF){
        if(v>=start && v<start+blocksize){
            v -= start; // make v in [0, blocksize)
            bit[v/int_len] |= 1<<(v%int_len);
        }
    }

    bool found = false;
    for(int i=0; i<blocksize/int_len+1; ++i){
        for(int j=0; j<int_len; ++j){
            if((bit[i] & (1<<j)) == 0){
                cout<<i*int_len+j+start<<endl;
                found = true;
                break;
            }
        }
        if(found) break;
    }

    delete[] block;
    delete[] bit;
    fclose(stdin);
    return 0;
}
//Q13.1.h


/*
Write a method to print the last K lines of an input file using C++.
*/

/*
一种方法是打开文件两次，第一次计算文件的行数N，第二次打开文件，跳过N-K行， 然后开始输出。
如果文件很大，这种方法的时间开销会非常大。

我们希望可以只打开文件一次，就可以输出文件中的最后k行。 我们可以开一个大小为k的字符串数组，然后将文件中的每一行循环读入
*/

#include <iostream>
#include <fstream>
using namespace std;

//we can use queue to simplify the code
void printLastKLines(ifstream &fin, int k){
    string line[k];
    int lines = 0;
    string tmp;
    while(getline(fin, tmp)){
        line[lines%k] = tmp;
        ++lines;
    }
    int start, cnt;
    if(lines < k){
        start = 0;
        cnt = lines;
    }
    else{
        start = lines%k;
        cnt = k;
    }
    for(int i=0; i<cnt; ++i)
        cout<<line[(start+i)%k]<<endl;
}
int main(){
    ifstream fin("13.1.in");
    int k = 4;
    printLastKLines(fin, k);
    fin.close();
    return 0;
}
//Q13.9.h


/*
smart pointer c++
*/

#include <iostream>
#include <cstdlib>
using namespace std;

template <typename T>
class SmartPointer{
public:
    SmartPointer(T* ptr){
        ref = ptr;
        ref_count = (unsigned*)malloc(sizeof(unsigned));
        *ref_count = 1;
    }
    
    SmartPointer(SmartPointer<T> &sptr){
        ref = sptr.ref;
        ref_count = sptr.ref_count;
        ++*ref_count;
    }
    
    SmartPointer<T>& operator=(SmartPointer<T> &sptr){
        if (this != &sptr) {
            if (--*ref_count == 0){
                clear();
                cout<<"operator= clear"<<endl;
            }
            
            ref = sptr.ref;
            ref_count = sptr.ref_count;
            ++*ref_count;
        }
        return *this;
    }
    
    ~SmartPointer(){
        if (--*ref_count == 0){
            clear();
            cout<<"destructor clear"<<endl;
        }
    }
    
    T getValue() { return *ref; }
    
private:
    void clear(){
        delete ref;
        free(ref_count);
        ref = NULL; // 避免它成为迷途指针
        ref_count = NULL;
    }
   
protected:  
    T *ref;
    unsigned *ref_count;
};

int main(){
    int *ip1 = new int();
    *ip1 = 11111;
    int *ip2 = new int();
    *ip2 = 22222;
    SmartPointer<int> sp1(ip1), sp2(ip2);
    SmartPointer<int> spa = sp1;
    sp2 = spa; // 注释掉它将得到不同输出
    return 0;
}
//Q13.10.h


/*
write an aligned malloc and free function
which supports allocating memory such that memory is divisible by a specific power of two
*/

void* alloc(size_t byes, size_t align){
	void* orig;
	void** aligned;
	int offset = align-1+sizeof(void*);
	orig = malloc(bytes+offset);
	aligned = (void**)( ((size_t)orig+offset) & (~(align-1)) );
	aligned[-1]=orig;
	return aligned;
}

void free(void* aligned){
	void* orig = ((void**)aligned)[-1];
	free(orig);
}
//Q16.3.h


/*
Implement a singleton design pattern as a template such that, for any given class Foo, 
you can call Singleton::instance() and get a pointer to an instance of a singleton of type Foo. 
Assume the existence of a class Lock which has acquire() and release() methods. 
How could you make your implementation thread safe and exception safe?
*/

#include <iostream>
using namespace std;

/* 线程同步锁 */
class Lock {
public:
    Lock() { /* 构造锁 */ }
    ~Lock() { /* 析构锁 */ }
    void AcquireLock() { /* 加锁操作 */ }
    void ReleaseLock() { /* 解锁操作 */ }
};

// 单例模式模板，只实例化一次
template <typename T>
class Singleton{
private:
    static Lock lock;
    static T* object;
protected:
    Singleton() { };
public:
    static T* Instance();
};

template <typename T>
Lock Singleton<T>::lock;

template <typename T>
T* Singleton<T>::object = NULL;

template <typename T>
T* Singleton<T>::Instance(){
    if (object == NULL){// 如果object未初始化，加锁初始化
        lock.AcquireLock();
        //这里再判断一次，因为多个线程可能同时通过第一个if
        //只有第一个线程去实例化object，之后object非NULL
        //后面的线程不再实例化它
        if (object == NULL){
            object = new T;
        }
        lock.ReleaseLock();
    }
    return object;
}
class Foo{
    
};
int main(){
    Foo* singleton_foo = Singleton<Foo>::Instance();
    return 0;
}
//Q16.4.h


/*
Suppose we have the following code:

class Foo{
public:
    A(.....); If A is called, a new thread will be created and
               the corresponding function will be executed. 
    B(.....); same as above 
    C(.....); same as above 
};
Foo f;
f.A(.....);
f.B(.....);
f.C(.....);
i) Can you design a mechanism to make sure that B is executed after A, and C is executed after B?

ii) Suppose we have the following code to use class Foo. We do not know how the threads will be scheduled in the OS.

Foo f;
f.A(.....); f.B(.....); f.C(.....); 
f.A(.....); f.B(.....); f.C(.....);
*/

第一问，初始的两个信号量都为0，函数A执行完后，信号量s_a会加1，这时B才可执行。 B执行完后信号量s_b加1，这时C才可执行。
以此保证A，B，C的执行顺序。 注意到函数A其实没有受到限制，所以A可以被多个线程多次执行。比如A执行3次， 
此时s_a=3；然后执行B，s_a=2,s_b=1；然后执行C，s_a=2,s_b=0； 然后执行B，s_a=1,s_b=1。即可以出现类似这种序列：AAABCB。

Semaphore s_a(0);
Semaphore s_b(0);
A {
    /***/
    s_a.release(1); // 信号量s_a加1
}
B {
    s_a.acquire(1); // 信号量s_a减1
    /****/
    s_b.release(1); // 信号量s_b加1
}
C {
    s_b.acquire(1); // 信号量s_b减1
    /******/
}
第二问代码如下，与第一问不同，以下代码可以确保执行顺序一定严格按照： ABCABCABC…进行。
因为每一时刻都只有一个信号量不为0， 且B中要获取的信号量在A中释放，C中要获取的信号量在B中释放，
 A中要获取的信号量在C中释放。这个保证了执行顺序一定是ABC。

Semaphore s_a(0);
Semaphore s_b(0);
Semaphore s_c(1);
A {
    s_c.acquire(1);
    /***/
    s_a.release(1);
}
B {
    s_a.acquire(1);
    /****/
    s_b.release(1);
}
C {
    s_b.acquire(1);
    /******/
    s_c.release(1);
}
//Q17.1.h


/*
Write a function to swap a number in place without temporary variables.
*/

// 实现1
void swap(int &a, int &b){
    b = a - b;
    a = a - b;
    b = a + b;
}
// 实现2
void swap(int &a, int &b){
    a = a ^ b;
    b = a ^ b;
    a = a ^ b;
}
/*
以上的swap函数，尤其是第2个实现，简洁美观高效，乃居家旅行必备良品。但是， 使用它们之前一定要想一想，你的程序中，是否有可能会让swap中的两个形参引用同一变量。 如果是，那么上述两个swap函数都将出问题。有人说，谁那么无聊去swap同一个变量。 那可不好说，比如你在操作一个数组中的元素，然后用到了以下语句：

swap(a[i], a[j]); // i==j时，出问题
你并没有注意到swap会去操作同一变量，可是当i等于j时，就相当于你这么干了。 然后呢，上面两个实现执行完第一条语句后，操作的那个内存中的数就变成0了。 后面的语句不会起到什么实际作用。

所以如果程序中有可能让swap函数去操作同一变量，就老老实实用最朴素的版本
*/
//Q17.3.h


/*
Write an algorithm which computes the number of trailing zeros in n factorial.
*/

/*
n阶乘末尾的0来自因子5和2相乘，5*2=10。因此，我们只需要计算n的阶乘里， 有多少对5和2。
注意到2出现的频率比5多，因此，我们只需要计算有多少个因子5即可

*/

int NumZeros(int n){
    if(n < 0) return -1;
    int num = 0;
    while((n /= 5) > 0){
        num += n;
    }
    return num;
}

/*
5!, 包含1*5, 1个5
10!, 包含1*5,2*5, 2个5
15!, 包含1*5,2*5,3*5, 3个5
20!, 包含1*5,2*5,3*5,4*5, 4个5
25!, 包含1*5,2*5,3*5,4*5,5*5, 6个5
...
给定一个n，用n除以5，得到的是从1到n中包含1个5的数的个数；然后用n除以5去更新n， 
相当于把每一个包含5的数中的因子5取出来一个。然后继续同样的操作，让n除以5， 将得到此时仍包含有5的数的个数，
依次类推。最后把计算出来的个数相加即可。 比如计算25的阶乘中末尾有几个0， 先用25除以5得到5，
表示我们从5,10,15,20,25中各拿一个因子5出来，总共拿了5个。 
更新n=25/5=5，再用n除以5得到1，表示我们从25中拿出另一个因子5， 其它的在第一次除以5后就不再包含因子5了。
*/
//Q17.4.h


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






//Q17.6.h


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





















//Q17.7.h


/*
translate a integer into english phrase

hint:
convert(19,323,984) =  convert(19) + "million"
					  +convert(323) + "thousand"
					  +convert(984) + ""
					  
convert(984) = digits[9] + "hundred" + tens[8] + digit[4]
convert(19) = teens[9]

*/
//Q17.11.h


/*
Write a method to generate a random number between 0 and 6, 
given a method that generates a random number between 0 and 4 (i.e., implement rand7() using rand5()).
*/

/*
如果a > b，那么一定可以用Randa去实现Randb。其中， Randa表示等概率生成1到a的函数，Randb表示等概率生成1到b的函数

// a > b
int Randb(){
    int x = INT_MAX
    while(x > b)
        x = Randa();
    return x;
}
*/

/*
wrong answer
Rand5() + Rand5()
上述代码可以生成1到9的数，但它们是等概率生成的吗？不是。生成1只有一种组合： 两个Rand5()都生成1时：(1, 1)；
而生成2有两种：(1, 2)和(2, 1)；生成6更多。 它们的生成是不等概率.
*/

/*
correct one
*/

int Rand7(){
    int x = INT_MAX;
    while(x > 21)
        x = 5 * Rand5() + Rand5() // Rand25 (0-24)
    return x%7 ;
}
//Q17.14.h


/*
Write a method to generate a random number between 0 and 6, 
given a method that generates a random number between 0 and 4 (i.e., implement rand7() using rand5()).
*/

/*
如果a > b，那么一定可以用Randa去实现Randb。其中， Randa表示等概率生成1到a的函数，Randb表示等概率生成1到b的函数

// a > b
int Randb(){
    int x = INT_MAX
    while(x > b)
        x = Randa();
    return x;
}
*/

/*
wrong answer
Rand5() + Rand5()
上述代码可以生成1到9的数，但它们是等概率生成的吗？不是。生成1只有一种组合： 两个Rand5()都生成1时：(1, 1)；
而生成2有两种：(1, 2)和(2, 1)；生成6更多。 它们的生成是不等概率.
*/

/*
correct one
*/

int Rand7(){
    int x = INT_MAX;
    while(x > 21)
        x = 5 * Rand5() + Rand5() // Rand25 (0-24)
    return x%7 ;
}
//Q17.15.h


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
//Q18.1.h


/*
Write a function that adds two numbers. You should not use + or any arithmetic operators.
*/

/*
为了解决这个问题，让我们来深入地思考一下，我们是如何去加两个数的。为了易于理解， 我们考虑10进制的情况。比如我们要算759 + 674，我们通常从最低位开始加， 考虑进位；然后加第二位，考虑进位…对于二进制，我们可以使用相同的方法， 每一位求和，然后考虑进位。

能把这个过程弄得更简单点吗？答案是YES，我们可以把求两个数的和分成两步， “加"与"进位"，看例子：

计算759 + 674，但不考虑进位，得到323。

计算759 + 674，只考虑进位，而不是去加每一位，得到1110。

把上面得到的两个数加起来(这里又要用到加，所以使用递归调用即可)

由于我们不能使用任何算术运算符，因此可供我们使用的就只有位运算符了。 于是我们把操作数看成二进制表示，然后对它们做类似的操作：

不考虑进位的按位求和，(0,0),(1,1)得0，(1,0),(0,1)得1， 使用异或操作可以满足要求。

只考虑进位，只有(1,1)才会产生进位，使用按位与可以满足要求。 当前位产生进位，要参与高一位的运算，因此按位与后要向左移动一位。

递归求和，直到进位为0

代码如下：
*/

//recursive
int Add2(int a, int b){
    if(b == 0) return a;
    int sum = a ^ b; // 各位相加，不计进位
    int carry = (a & b) << 1; // 记下进位
    return Add2(sum, carry); // 求sum和carry的和
}
//iterative
int Add3(int a, int b){
    while(b != 0){
        int sum = a ^ b;
        int carry = (a & b) << 1;
        a = sum;
        b = carry;
    }
    return a;
}
//let compiler do the work
int myAdd2(int a, int b){
	char* c = (char*)a;
	return (int)&c[b];
}
//Q18.2.h


/*
Write a method to shuffle a deck of cards. It must be a perfect shuffle - 
in other words, each 52! permutations of the deck has to be equally likely.
Assume that you are given a random number generator which is perfect
*/

/*
最直观的思路是什么？很简单，每次从牌堆中随机地拿一张出来。那么， 第一次拿有52种可能，拿完后剩下51张；
第二次拿有51种可能，第三次拿有50种可能， …，
一直这样随机地拿下去直到拿完最后1张，我们就从52!种可能中取出了一种排列， 这个排列对应的概率是1/(52!)，
正好是题目所要求的。
*/

#include <iostream>
#include <cstdlib>
using namespace std;

void Swap(int &a, int &b){// 有可能swap同一变量，不能用异或版本!!!!!!!!!
    int t = a;
    a = b;
    b = t;
}

//O(N)
void RandomShuffle(int a[], int n){
    for(int i=0; i<n; ++i){
        int j = rand() % (n-i) + i;// 产生 i 到 n-1 间的随机数
        Swap(a[i], a[j]);
    }
}
int main(){
    srand((unsigned)time(0));
    int n = 9;
    int a[] = {
        1, 2, 3, 4, 5, 6, 7, 8, 9
    };
    RandomShuffle(a, n);
    for(int i=0; i<n; ++i)
        cout<<a[i]<<endl;
    return 0;
}

/*
similar problem
Write a method to randomly generate a set of m integers from an array of size n. 
Each element must have equal probability of being chosen.

选第1个元素：在n个中随机选，因此概率为1/n

选第2个元素：在剩下的n-1个中随机选：1/(n-1)，由于第1次没有选中它， 而是在另外n-1个中选：(n-1)/n，因此概率为：(n-1)/n * 1/(n-1) = 1/n

选第3个元素：同上：(n-1)/n * (n-2)/(n-1) * 1/(n-2) = 1/n
*/
void PickMRandomly(int a[], int n, int m){
    for(int i=0; i<m; ++i){
        int j = rand() % (n-i) + i;// 产生i到n-1间的随机数
        Swap(a[i], a[j]);
    }
}


//Q18.4.h


/*
Write a method to count the number of 2s between 0 and n.
input 25
output 9(2,12,20,21,22,23,24,25)
*/

//naive
int Count2(int n){
    int count = 0;
    while(n > 0){
        if(n%10 == 2)
            ++count;
        n /= 10;
    }
    return count;
}

int Count2s1(int n){
    int count = 0;
    for(int i=0; i<=n; ++i)
        count += Count2(i);
    return count;
}

//编程之美
/*
假设一个5位数N=abcde，我们现在来考虑百位上出现2的次数，即，从0到abcde的数中， 有多少个数的百位上是2。
分析完它，就可以用同样的方法去计算个位，十位，千位， 万位等各个位上出现2的次数

当某一位的数字小于2时，那么该位出现2的次数为：更高位数字x当前位数
当某一位的数字等于2时，那么该位出现2的次数为：更高位数字x当前位数+低位数字+1
当某一位的数字大于2时，那么该位出现2的次数为：(更高位数字+1)x当前位数

*/

int Count2s(int n){
    int count = 0;
    int factor = 1;
    int low = 0, cur = 0, high = 0;

    while(n/factor != 0){
        low = n - (n/factor) * factor;//低位数字
        cur = (n/factor) % 10;//当前位数字
        high = n / (factor*10);//高位数字

        switch(cur){
        case 0:
        case 1:
            count += high * factor;
            break;
        case 2:
            count += high * factor + low + 1;
            break;
        default:
            count += (high + 1) * factor;
            break;
        }

        factor *= 10;
    }

    return count;
}

/*general case
当某一位的数字小于i时，那么该位出现i的次数为：更高位数字x当前位数
当某一位的数字等于i时，那么该位出现i的次数为：更高位数字x当前位数+低位数字+1
当某一位的数字大于i时，那么该位出现i的次数为：(更高位数字+1)x当前位数
代码如下：
*/

int Countis(int n, int i){
    if(i<1 || i>9) return -1;//i只能是1到9

    int count = 0;
    int factor = 1;
    int low = 0, cur = 0, high = 0;

    while(n/factor != 0){
        low = n - (n/factor) * factor;//低位数字
        cur = (n/factor) % 10;//当前位数字
        high = n / (factor*10);//高位数字

        if(cur < i)
            count += high * factor;
        else if(cur == i)
            count += high * factor + low + 1;
        else
            count += (high + 1) * factor;

        factor *= 10;
    }

    return count;
}



//Q18.7.h


/*
Write a program to find the longest word made of other words.

Example:
input: cat, banana, dog, nana, walk, walker, dogwalker
Output: dogwalker 
*/

	public static void findLongestWord(String[] strs){
        Set<String> dict = new HashSet<String>();
        for(String s: strs) dict.add(s); //cache
        Comparator<String> mycomp = new Comparator<String>(){
            @Override
            public int compare(String a, String b){
                if(a.length()<b.length()) return 1;
                else if(a.length() == b.length()) return 0;
                else return -1;
            }
        };
        Arrays.sort(strs, mycomp);
        for(String s: strs){
            dict.remove(s);
            if(dfs(dict,s))
                System.out.println(s);//this will print all words that can be combined from other
            dict.add(s);
        }
    }
    public static boolean dfs(Set<String> dict, String target){
        if(dict.contains(target)) return true;
        for(int i = 1;i<target.length();i++){
            if(dict.contains(target.substring(0,i))&&dfs(dict,target.substring(i)))
                return true;
        }
        return false;
    }
//Q18.8.h


/*
Given a string s and an array of smaller strings T, design a method to search s for each small string in T.
*/

/*
KMP

我们把S称为目标串，T中的字符串称为模式串。
设目标串S的长度为m，模式串的平均长度为 n，共有k个模式串。
如果我们用KMP算法,匹配一个模式串和目标串的时间为O(m+n)，所以总时间复杂度为：O(k(m+n))。 
模式串则是一些较短的字符串，也就是m一般要远大于n。 
这时候如果我们要匹配的模式串非常多(即k非常大)，那么我们使用上述算法就会非常慢。 
这也是为什么KMP或BM一般只用于单模式匹配，而不用于多模式匹配。
*/

/*
suffix tree

假设字符串S = “abcd"，那么它的所有后缀是：
abcd
bcd
cd
d
我们发现，如果一个串t是S的子串，那么t一定是S某个后缀的前缀。比如t = bc， 
那么它是后缀bcd的前缀；又比如说t = c，那么它是后缀cd的前缀。
*/

//上述方法总的时间复杂度是：O(m2 + kn)，有没更快的算法呢？答案是肯定的.

// Ukkonen算法， 该算法可以在线性时间和空间下构造后缀树，而且它还是在线算法


#include<iostream>
#include<vector>
#include<map>
#include<set>
#include<queue>
#include <limits>
#include <cassert>
#include <cmath>
#include <sstream>

using namespace std;

class Trie{

public:

    struct TrieNode{

        TrieNode():c('\0'),isWord(false){}
        TrieNode(char v):c(v),isWord(false){}
        bool find(char c){ return childs.count(c)!=0 ; }
        TrieNode* insert(char c){ childs[c] = new TrieNode(c) ; return childs[c];}
        TrieNode* operator[] (char c){ return childs[c];}
        bool isLeaf(){ return childs.size()==0; }

        char c;
        bool isWord;
        map<char,TrieNode*> childs;
    };

    Trie(){root = new TrieNode();}

    void insert(const string& str);
    bool search(const string& str);
    void getHints(vector<string>* v, const string& prefix);

private:
    TrieNode* root;
    void trieDfs(vector<string>* v, TrieNode* root, string preStr);
};

void Trie::insert(const string& str){
    TrieNode* curNode = root;
    string curStr=str;
    while(curStr!=""){
        char c = curStr[0];
        if( !curNode->find(c) ){
            curNode->insert(c);
        }
        curNode=(*curNode)[c];
        curStr=curStr.substr(1);
    }
    curNode->isWord=true;
}

bool Trie::search(const string& str){
    TrieNode* curNode = root;
    string curStr=str;
    while(curStr!=""){
        char c = curStr[0];
        if( !curNode->find(c) ){
            return false;
        }
        curNode=(*curNode)[c];
        curStr=curStr.substr(1);
    }
    return curNode->isWord;
}

void Trie::getHints(vector<string>* v, const string& prefix){
    TrieNode* curNode = root;
    string curStr=prefix;
    while(curStr!=""){
        char c = curStr[0];
        if( !curNode->find(c) ){
            return ;
        }
        curNode=(*curNode)[c];
        curStr=curStr.substr(1);
    }

    trieDfs(v,curNode,prefix);
}

void Trie::trieDfs(vector<string>* v, TrieNode* root, string str){

    if( root->isLeaf() ){
        v->push_back(str);
    }
    else{
        if(root->isWord)
            v->push_back(str);

        map<char,TrieNode*>::iterator iter = root->childs.begin();
        while( iter != root->childs.end() ){
            trieDfs(v, iter->second,str+iter->first);
            ++iter;
        }
    }
}

void algo(){
    Trie trie;
    trie.insert("Ying");
    trie.insert("Zhan");
    trie.insert("Yan");
    trie.insert("Xingwei");
    trie.insert("XinJiang");
    trie.insert("YY");
    trie.insert("Yellow");
    trie.insert("Yes");

    if( trie.search("Ying") )
        cout <<"Ying is found";
    else
        cout <<"Ying is not found";
    cout<<endl;

    if( trie.search("Yin") )
        cout <<"Yin is found";
    else
        cout <<"Yin is not found";
    cout<<endl;

    if( trie.search("Xingwei") )
        cout <<"Xingwei is found";
    else
        cout <<"Xingwei is not found";
    cout<<endl;

    {
        cout<<endl<<"Hints(Y):"<<endl<<endl;
        vector<string> hints;
        trie.getHints(&hints,"Y");
        for(int i = 0; i< hints.size(); ++i){
            cout << hints[i] << endl;
        }
    }

    {
        cout<<endl<<"Hints(Xi):"<<endl<<endl;
        vector<string> hints;
        trie.getHints(&hints,"Xi");
        for(int i = 0; i< hints.size(); ++i){
            cout << hints[i] << endl;
        }
    }
}
//Q18.9.h


/*
online median
Numbers are randomly generated and passed to a method. 
Write a program to find and maintain the median value as new values are generated.
*/

#include<set>
#include<queue>
#include <limits>
#include <cassert>
#include <cmath>
#include <sstream>
//#include <unordered_map>

using namespace std;

void online_median(istringstream* sin) {
  // Min-heap stores the bigger part of the stream.
  priority_queue<int, vector<int>, greater<int> > H;
  // Max-heap stores the smaller part of the stream.
  priority_queue<int, vector<int>, less<int> > L;

  int x;
  while (*sin >> x) {

    if (!L.empty() && x > L.top()) {
      H.push(x);
    } else {
      L.push(x);
    }

    //balance
    if (H.size() > L.size() + 1) {
      L.push(H.top());
      H.pop();
    } else if (L.size() > H.size() + 1) {
      H.push(L.top());
      L.pop();
    }

    if (H.size() == L.size()) {
      cout << 0.5 * (H.top() + L.top()) << endl;
    } else {
      cout << (H.size() > L.size() ? H.top() : L.top()) << endl;
    }
  }
}
//Q18.10.h


/*
word ladder
*/

/*
Given two words (start and end), and a dictionary, find all shortest transformation sequence(s) from start to end, such that:

Only one letter can be changed at a time
Each intermediate word must exist in the dictionary
For example,

Given:
start = "hit"
end = "cog"
dict = ["hot","dot","dog","lot","log"]
Return
  [
    ["hit","hot","dot","dog","cog"],
    ["hit","hot","lot","log","cog"]
  ]
Note:
All words have the same length.
All words contain only lowercase alphabetic characters.
*/

//BFS to generate the graph then DFS to find pathes
class Solution {
public:

    vector<string > getNeighbors(const string& s,  unordered_set<string> &dict){
        vector<string > neighbors;
        string start(s);
        for(int i = 0; i < start.size(); ++i){
            for(int j = 'a' ; j <= 'z' ; ++j){
                start[i] = j;
                if(start!=s && dict.count(start) > 0){
                    neighbors.push_back(start);
                }
            }
            start = s;
        }
        return neighbors;
    }
    
    vector<vector<string> > res;
    vector<vector<string>> findLadders(string start, string end, unordered_set<string> &dict) {
        unordered_map<string, vector<string> > graph;//build graph from start to end
        unordered_set<string> visited; //track visited
        unordered_set<string> curr, prev; // bfs levels
        prev.insert(start);
        visited.insert(start);
        //BFS to build graph
        while(!prev.empty()){
            //mark prev visited
            for(unordered_set<string>::iterator iter = prev.begin(); iter != prev.end(); ++iter){
                visited.insert(*iter);
            }
            //get curr level
            for(unordered_set<string>::iterator iter = prev.begin(); iter != prev.end(); ++iter){
                const string& preStr = *iter;
                vector<string> neighbors = getNeighbors(preStr,dict);
                for(int i = 0; i < neighbors.size(); ++i){
                    string& curStr = neighbors[i];
                    if(visited.count(curStr)==0){ // not visited before
                        curr.insert(curStr); 
                        graph[preStr].push_back(curStr);
                        //visited.insert(curStr);//Don't mark visited here, otherwise would block other paths
                    }
                }
            }
            if(curr.size()==0) return res; //not found
            if(curr.count(end)>0) break; //found
            prev = curr;
            curr.clear();
        }
        
        //DFS to find all paths
        vector<string> path;
        getPath(start, end, graph, path);
        return res;
    }
    
    void getPath(string& start, string& end, unordered_map<string,vector<string> >& graph, vector<string> & path) {
        path.push_back(start);
        if (start == end) {
            res.push_back(vector<string>(path.begin(), path.end()));
        }
        else {
            vector<string>& childs = graph[start];
            for (int i = 0; i < childs.size(); ++i  ) {
                getPath(childs[i], end, graph, path);
            }
        }
        path.pop_back();
    }
};
//Q18.11.h


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

//Q18.12.h


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
