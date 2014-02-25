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