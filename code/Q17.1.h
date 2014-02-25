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