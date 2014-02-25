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