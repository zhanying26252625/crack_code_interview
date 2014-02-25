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