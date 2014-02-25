/*
Write a method to count the number of 2s between 0 and n.
input 25
output 9(2,12,20,21,22,23,24,25)
*/

//naive
int Count2(int n){
    int count = 0;
    while(n > 0){
        if(n%10 == 2)
            ++count;
        n /= 10;
    }
    return count;
}

int Count2s1(int n){
    int count = 0;
    for(int i=0; i<=n; ++i)
        count += Count2(i);
    return count;
}

//编程之美
/*
假设一个5位数N=abcde，我们现在来考虑百位上出现2的次数，即，从0到abcde的数中， 有多少个数的百位上是2。
分析完它，就可以用同样的方法去计算个位，十位，千位， 万位等各个位上出现2的次数

当某一位的数字小于2时，那么该位出现2的次数为：更高位数字x当前位数
当某一位的数字等于2时，那么该位出现2的次数为：更高位数字x当前位数+低位数字+1
当某一位的数字大于2时，那么该位出现2的次数为：(更高位数字+1)x当前位数

*/

int Count2s(int n){
    int count = 0;
    int factor = 1;
    int low = 0, cur = 0, high = 0;

    while(n/factor != 0){
        low = n - (n/factor) * factor;//低位数字
        cur = (n/factor) % 10;//当前位数字
        high = n / (factor*10);//高位数字

        switch(cur){
        case 0:
        case 1:
            count += high * factor;
            break;
        case 2:
            count += high * factor + low + 1;
            break;
        default:
            count += (high + 1) * factor;
            break;
        }

        factor *= 10;
    }

    return count;
}

/*general case
当某一位的数字小于i时，那么该位出现i的次数为：更高位数字x当前位数
当某一位的数字等于i时，那么该位出现i的次数为：更高位数字x当前位数+低位数字+1
当某一位的数字大于i时，那么该位出现i的次数为：(更高位数字+1)x当前位数
代码如下：
*/

int Countis(int n, int i){
    if(i<1 || i>9) return -1;//i只能是1到9

    int count = 0;
    int factor = 1;
    int low = 0, cur = 0, high = 0;

    while(n/factor != 0){
        low = n - (n/factor) * factor;//低位数字
        cur = (n/factor) % 10;//当前位数字
        high = n / (factor*10);//高位数字

        if(cur < i)
            count += high * factor;
        else if(cur == i)
            count += high * factor + low + 1;
        else
            count += (high + 1) * factor;

        factor *= 10;
    }

    return count;
}


