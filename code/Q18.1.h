/*
Write a function that adds two numbers. You should not use + or any arithmetic operators.
*/

/*
为了解决这个问题，让我们来深入地思考一下，我们是如何去加两个数的。为了易于理解， 我们考虑10进制的情况。比如我们要算759 + 674，我们通常从最低位开始加， 考虑进位；然后加第二位，考虑进位…对于二进制，我们可以使用相同的方法， 每一位求和，然后考虑进位。

能把这个过程弄得更简单点吗？答案是YES，我们可以把求两个数的和分成两步， “加"与"进位"，看例子：

计算759 + 674，但不考虑进位，得到323。

计算759 + 674，只考虑进位，而不是去加每一位，得到1110。

把上面得到的两个数加起来(这里又要用到加，所以使用递归调用即可)

由于我们不能使用任何算术运算符，因此可供我们使用的就只有位运算符了。 于是我们把操作数看成二进制表示，然后对它们做类似的操作：

不考虑进位的按位求和，(0,0),(1,1)得0，(1,0),(0,1)得1， 使用异或操作可以满足要求。

只考虑进位，只有(1,1)才会产生进位，使用按位与可以满足要求。 当前位产生进位，要参与高一位的运算，因此按位与后要向左移动一位。

递归求和，直到进位为0

代码如下：
*/

//recursive
int Add2(int a, int b){
    if(b == 0) return a;
    int sum = a ^ b; // 各位相加，不计进位
    int carry = (a & b) << 1; // 记下进位
    return Add2(sum, carry); // 求sum和carry的和
}
//iterative
int Add3(int a, int b){
    while(b != 0){
        int sum = a ^ b;
        int carry = (a & b) << 1;
        a = sum;
        b = carry;
    }
    return a;
}
//let compiler do the work
int myAdd2(int a, int b){
	char* c = (char*)a;
	return (int)&c[b];
}