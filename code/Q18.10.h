/*
word ladder
*/

/*
Given two words (start and end), and a dictionary, find all shortest transformation sequence(s) from start to end, such that:

Only one letter can be changed at a time
Each intermediate word must exist in the dictionary
For example,

Given:
start = "hit"
end = "cog"
dict = ["hot","dot","dog","lot","log"]
Return
  [
    ["hit","hot","dot","dog","cog"],
    ["hit","hot","lot","log","cog"]
  ]
Note:
All words have the same length.
All words contain only lowercase alphabetic characters.
*/

//BFS to generate the graph then DFS to find pathes
class Solution {
public:

    vector<string > getNeighbors(const string& s,  unordered_set<string> &dict){
        vector<string > neighbors;
        string start(s);
        for(int i = 0; i < start.size(); ++i){
            for(int j = 'a' ; j <= 'z' ; ++j){
                start[i] = j;
                if(start!=s && dict.count(start) > 0){
                    neighbors.push_back(start);
                }
            }
            start = s;
        }
        return neighbors;
    }
    
    vector<vector<string> > res;
    vector<vector<string>> findLadders(string start, string end, unordered_set<string> &dict) {
        unordered_map<string, vector<string> > graph;//build graph from start to end
        unordered_set<string> visited; //track visited
        unordered_set<string> curr, prev; // bfs levels
        prev.insert(start);
        visited.insert(start);
        //BFS to build graph
        while(!prev.empty()){
            //mark prev visited
            for(unordered_set<string>::iterator iter = prev.begin(); iter != prev.end(); ++iter){
                visited.insert(*iter);
            }
            //get curr level
            for(unordered_set<string>::iterator iter = prev.begin(); iter != prev.end(); ++iter){
                const string& preStr = *iter;
                vector<string> neighbors = getNeighbors(preStr,dict);
                for(int i = 0; i < neighbors.size(); ++i){
                    string& curStr = neighbors[i];
                    if(visited.count(curStr)==0){ // not visited before
                        curr.insert(curStr); 
                        graph[preStr].push_back(curStr);
                        //visited.insert(curStr);//Don't mark visited here, otherwise would block other paths
                    }
                }
            }
            if(curr.size()==0) return res; //not found
            if(curr.count(end)>0) break; //found
            prev = curr;
            curr.clear();
        }
        
        //DFS to find all paths
        vector<string> path;
        getPath(start, end, graph, path);
        return res;
    }
    
    void getPath(string& start, string& end, unordered_map<string,vector<string> >& graph, vector<string> & path) {
        path.push_back(start);
        if (start == end) {
            res.push_back(vector<string>(path.begin(), path.end()));
        }
        else {
            vector<string>& childs = graph[start];
            for (int i = 0; i < childs.size(); ++i  ) {
                getPath(childs[i], end, graph, path);
            }
        }
        path.pop_back();
    }
};