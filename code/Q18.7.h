/*
Write a program to find the longest word made of other words.

Example:
input: cat, banana, dog, nana, walk, walker, dogwalker
Output: dogwalker 
*/

	public static void findLongestWord(String[] strs){
        Set<String> dict = new HashSet<String>();
        for(String s: strs) dict.add(s); //cache
        Comparator<String> mycomp = new Comparator<String>(){
            @Override
            public int compare(String a, String b){
                if(a.length()<b.length()) return 1;
                else if(a.length() == b.length()) return 0;
                else return -1;
            }
        };
        Arrays.sort(strs, mycomp);
        for(String s: strs){
            dict.remove(s);
            if(dfs(dict,s))
                System.out.println(s);//this will print all words that can be combined from other
            dict.add(s);
        }
    }
    public static boolean dfs(Set<String> dict, String target){
        if(dict.contains(target)) return true;
        for(int i = 1;i<target.length();i++){
            if(dict.contains(target.substring(0,i))&&dfs(dict,target.substring(i)))
                return true;
        }
        return false;
    }