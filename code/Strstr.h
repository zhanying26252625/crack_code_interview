/*
Implement strStr().

Returns a pointer to the first occurrence of needle in haystack, or null if needle is not part of haystack.
*/

class Solution {
public:
    char * strStr(char * haystack, char * needle) {
        if (!haystack || !needle) return haystack;
        int n = strlen(haystack);
        int m = strlen(needle);
        if (n < m) return NULL;
        if (m == 0) return haystack;
        return strStr3(haystack, n, needle, m);
    }

    // brute force, takes O(n*m) time
    char *strStr1(char *haystack, int n, char *needle, int m) {
        int i = 0;
        while (i < n-m+1) {
            int j = 0;
            while (j < m && haystack[i] == needle[j]) {
                i++, j++;
            }
            if (j == m) return haystack+(i-j);
            i = i-j+1;
        }
        return NULL;
    }
    
    /*
    // Rabin-Karp (RK), takes O(n+m) times, but O(n*m) worst case
    char * strStr2(char * haystack, int n, char * needle, int m) {
        int hn = 0;
        for (int i = 0; i < m; i++) hn = mod(hn*B+needle[i], M);

        int hh = 0;
        for (int i = 0; i < m; i++) hh = mod(hh*B+haystack[i], M);

        if (hh == hn) return haystack;

        int E = 1;
        for (int i = 1; i < m; i++) E = mod(E*B, M);

        for (int i = m; i < n; i++) {
            hh = mod(hh-mod(haystack[i-m]*E, M), M);
            hh = mod(hh*B+haystack[i], M);

            if (hh == hn) {
                int j = 0;
                while (j < m && haystack[i+j] == needle[j]) j++;
                return haystack+i-m+1;
            }
        }
        return NULL;
    }

    int mod(int a, int b) {
        return (a%b+b)%b;
    }
    */
    
    // Knuth-Morris-Pratt Algorithm (KMP), takes O(n+m)
    char *strStr(char *haystack, char *needle) {
        if (haystack == NULL || needle == NULL) return NULL;
        int N = strlen(haystack);
        int M = strlen(needle);
        if(M==0) return haystack;
        vector<int> overlay(M,-1);
        //initialize overlay array
        for(int i = 1; i < M; ++i){
            int pre = overlay[i-1];
            while(pre>=0&&needle[pre+1]!=needle[i])
                pre=overlay[pre];
            if(needle[pre+1]==needle[i]) overlay[i]=pre+1;
        }
        //search
        int i = 0, j = 0;
        while(i<N){
            if(haystack[i]==needle[j]){
                ++i; ++j;
                if(j==M) return haystack+i-M;
            }
            else{
                if(j==0) ++i;
                else j = overlay[j-1] + 1;
            }
        }
        return NULL;
    }
};