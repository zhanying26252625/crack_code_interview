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