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