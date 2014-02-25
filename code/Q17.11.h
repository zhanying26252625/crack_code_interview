/*
Write a method to generate a random number between 0 and 6, 
given a method that generates a random number between 0 and 4 (i.e., implement rand7() using rand5()).
*/

/*
如果a > b，那么一定可以用Randa去实现Randb。其中， Randa表示等概率生成1到a的函数，Randb表示等概率生成1到b的函数

// a > b
int Randb(){
    int x = INT_MAX
    while(x > b)
        x = Randa();
    return x;
}
*/

/*
wrong answer
Rand5() + Rand5()
上述代码可以生成1到9的数，但它们是等概率生成的吗？不是。生成1只有一种组合： 两个Rand5()都生成1时：(1, 1)；
而生成2有两种：(1, 2)和(2, 1)；生成6更多。 它们的生成是不等概率.
*/

/*
correct one
*/

int Rand7(){
    int x = INT_MAX;
    while(x > 21)
        x = 5 * Rand5() + Rand5() // Rand25 (0-24)
    return x%7 ;
}