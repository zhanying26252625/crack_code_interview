/*
Suppose we have the following code:

class Foo{
public:
    A(.....); If A is called, a new thread will be created and
               the corresponding function will be executed. 
    B(.....); same as above 
    C(.....); same as above 
};
Foo f;
f.A(.....);
f.B(.....);
f.C(.....);
i) Can you design a mechanism to make sure that B is executed after A, and C is executed after B?

ii) Suppose we have the following code to use class Foo. We do not know how the threads will be scheduled in the OS.

Foo f;
f.A(.....); f.B(.....); f.C(.....); 
f.A(.....); f.B(.....); f.C(.....);
*/

第一问，初始的两个信号量都为0，函数A执行完后，信号量s_a会加1，这时B才可执行。 B执行完后信号量s_b加1，这时C才可执行。
以此保证A，B，C的执行顺序。 注意到函数A其实没有受到限制，所以A可以被多个线程多次执行。比如A执行3次， 
此时s_a=3；然后执行B，s_a=2,s_b=1；然后执行C，s_a=2,s_b=0； 然后执行B，s_a=1,s_b=1。即可以出现类似这种序列：AAABCB。

Semaphore s_a(0);
Semaphore s_b(0);
A {
    /***/
    s_a.release(1); // 信号量s_a加1
}
B {
    s_a.acquire(1); // 信号量s_a减1
    /****/
    s_b.release(1); // 信号量s_b加1
}
C {
    s_b.acquire(1); // 信号量s_b减1
    /******/
}
第二问代码如下，与第一问不同，以下代码可以确保执行顺序一定严格按照： ABCABCABC…进行。
因为每一时刻都只有一个信号量不为0， 且B中要获取的信号量在A中释放，C中要获取的信号量在B中释放，
 A中要获取的信号量在C中释放。这个保证了执行顺序一定是ABC。

Semaphore s_a(0);
Semaphore s_b(0);
Semaphore s_c(1);
A {
    s_c.acquire(1);
    /***/
    s_a.release(1);
}
B {
    s_a.acquire(1);
    /****/
    s_b.release(1);
}
C {
    s_b.acquire(1);
    /******/
    s_c.release(1);
}