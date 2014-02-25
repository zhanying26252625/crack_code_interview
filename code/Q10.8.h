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