/*
Write a method to print the last K lines of an input file using C++.
*/

/*
一种方法是打开文件两次，第一次计算文件的行数N，第二次打开文件，跳过N-K行， 然后开始输出。
如果文件很大，这种方法的时间开销会非常大。

我们希望可以只打开文件一次，就可以输出文件中的最后k行。 我们可以开一个大小为k的字符串数组，然后将文件中的每一行循环读入
*/

#include <iostream>
#include <fstream>
using namespace std;

//we can use queue to simplify the code
void printLastKLines(ifstream &fin, int k){
    string line[k];
    int lines = 0;
    string tmp;
    while(getline(fin, tmp)){
        line[lines%k] = tmp;
        ++lines;
    }
    int start, cnt;
    if(lines < k){
        start = 0;
        cnt = lines;
    }
    else{
        start = lines%k;
        cnt = k;
    }
    for(int i=0; i<cnt; ++i)
        cout<<line[(start+i)%k]<<endl;
}
int main(){
    ifstream fin("13.1.in");
    int k = 4;
    printLastKLines(fin, k);
    fin.close();
    return 0;
}