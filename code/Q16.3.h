/*
Implement a singleton design pattern as a template such that, for any given class Foo, 
you can call Singleton::instance() and get a pointer to an instance of a singleton of type Foo. 
Assume the existence of a class Lock which has acquire() and release() methods. 
How could you make your implementation thread safe and exception safe?
*/

#include <iostream>
using namespace std;

/* 线程同步锁 */
class Lock {
public:
    Lock() { /* 构造锁 */ }
    ~Lock() { /* 析构锁 */ }
    void AcquireLock() { /* 加锁操作 */ }
    void ReleaseLock() { /* 解锁操作 */ }
};

// 单例模式模板，只实例化一次
template <typename T>
class Singleton{
private:
    static Lock lock;
    static T* object;
protected:
    Singleton() { };
public:
    static T* Instance();
};

template <typename T>
Lock Singleton<T>::lock;

template <typename T>
T* Singleton<T>::object = NULL;

template <typename T>
T* Singleton<T>::Instance(){
    if (object == NULL){// 如果object未初始化，加锁初始化
        lock.AcquireLock();
        //这里再判断一次，因为多个线程可能同时通过第一个if
        //只有第一个线程去实例化object，之后object非NULL
        //后面的线程不再实例化它
        if (object == NULL){
            object = new T;
        }
        lock.ReleaseLock();
    }
    return object;
}
class Foo{
    
};
int main(){
    Foo* singleton_foo = Singleton<Foo>::Instance();
    return 0;
}