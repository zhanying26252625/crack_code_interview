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