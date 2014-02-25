/*
Given a string s and an array of smaller strings T, design a method to search s for each small string in T.
*/

/*
KMP

我们把S称为目标串，T中的字符串称为模式串。
设目标串S的长度为m，模式串的平均长度为 n，共有k个模式串。
如果我们用KMP算法,匹配一个模式串和目标串的时间为O(m+n)，所以总时间复杂度为：O(k(m+n))。 
模式串则是一些较短的字符串，也就是m一般要远大于n。 
这时候如果我们要匹配的模式串非常多(即k非常大)，那么我们使用上述算法就会非常慢。 
这也是为什么KMP或BM一般只用于单模式匹配，而不用于多模式匹配。
*/

/*
suffix tree

假设字符串S = “abcd"，那么它的所有后缀是：
abcd
bcd
cd
d
我们发现，如果一个串t是S的子串，那么t一定是S某个后缀的前缀。比如t = bc， 
那么它是后缀bcd的前缀；又比如说t = c，那么它是后缀cd的前缀。
*/

//上述方法总的时间复杂度是：O(m2 + kn)，有没更快的算法呢？答案是肯定的.

// Ukkonen算法， 该算法可以在线性时间和空间下构造后缀树，而且它还是在线算法


#include<iostream>
#include<vector>
#include<map>
#include<set>
#include<queue>
#include <limits>
#include <cassert>
#include <cmath>
#include <sstream>

using namespace std;

class Trie{

public:

    struct TrieNode{

        TrieNode():c('\0'),isWord(false){}
        TrieNode(char v):c(v),isWord(false){}
        bool find(char c){ return childs.count(c)!=0 ; }
        TrieNode* insert(char c){ childs[c] = new TrieNode(c) ; return childs[c];}
        TrieNode* operator[] (char c){ return childs[c];}
        bool isLeaf(){ return childs.size()==0; }

        char c;
        bool isWord;
        map<char,TrieNode*> childs;
    };

    Trie(){root = new TrieNode();}

    void insert(const string& str);
    bool search(const string& str);
    void getHints(vector<string>* v, const string& prefix);

private:
    TrieNode* root;
    void trieDfs(vector<string>* v, TrieNode* root, string preStr);
};

void Trie::insert(const string& str){
    TrieNode* curNode = root;
    string curStr=str;
    while(curStr!=""){
        char c = curStr[0];
        if( !curNode->find(c) ){
            curNode->insert(c);
        }
        curNode=(*curNode)[c];
        curStr=curStr.substr(1);
    }
    curNode->isWord=true;
}

bool Trie::search(const string& str){
    TrieNode* curNode = root;
    string curStr=str;
    while(curStr!=""){
        char c = curStr[0];
        if( !curNode->find(c) ){
            return false;
        }
        curNode=(*curNode)[c];
        curStr=curStr.substr(1);
    }
    return curNode->isWord;
}

void Trie::getHints(vector<string>* v, const string& prefix){
    TrieNode* curNode = root;
    string curStr=prefix;
    while(curStr!=""){
        char c = curStr[0];
        if( !curNode->find(c) ){
            return ;
        }
        curNode=(*curNode)[c];
        curStr=curStr.substr(1);
    }

    trieDfs(v,curNode,prefix);
}

void Trie::trieDfs(vector<string>* v, TrieNode* root, string str){

    if( root->isLeaf() ){
        v->push_back(str);
    }
    else{
        if(root->isWord)
            v->push_back(str);

        map<char,TrieNode*>::iterator iter = root->childs.begin();
        while( iter != root->childs.end() ){
            trieDfs(v, iter->second,str+iter->first);
            ++iter;
        }
    }
}

void algo(){
    Trie trie;
    trie.insert("Ying");
    trie.insert("Zhan");
    trie.insert("Yan");
    trie.insert("Xingwei");
    trie.insert("XinJiang");
    trie.insert("YY");
    trie.insert("Yellow");
    trie.insert("Yes");

    if( trie.search("Ying") )
        cout <<"Ying is found";
    else
        cout <<"Ying is not found";
    cout<<endl;

    if( trie.search("Yin") )
        cout <<"Yin is found";
    else
        cout <<"Yin is not found";
    cout<<endl;

    if( trie.search("Xingwei") )
        cout <<"Xingwei is found";
    else
        cout <<"Xingwei is not found";
    cout<<endl;

    {
        cout<<endl<<"Hints(Y):"<<endl<<endl;
        vector<string> hints;
        trie.getHints(&hints,"Y");
        for(int i = 0; i< hints.size(); ++i){
            cout << hints[i] << endl;
        }
    }

    {
        cout<<endl<<"Hints(Xi):"<<endl<<endl;
        vector<string> hints;
        trie.getHints(&hints,"Xi");
        for(int i = 0; i< hints.size(); ++i){
            cout << hints[i] << endl;
        }
    }
}