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