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