/*
find number of ways to parenthesize a boolean expression(&,^,|) to achieve a given result
*/

/*
algo, breakdown each boolean operation

f(e1|e2,true) = f(e1,true) * f(e2,true) + f(e1,false) * f(e2,true) + f(e1,true) * f(e2,false)
etc

cache intermediate result.
key could be expr+":"+result for f(expr,result)
*/