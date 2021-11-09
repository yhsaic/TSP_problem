# TSP_problem
# currently files

1. a data structure X for a possible solution, with a function that can validate whether the solution has consistent information and that throws an exception otherwise
2. a way to read an object of X from a text file (that also first validates the instance with the function from point 1)
3. a way to write an object of X to a text file (compatible with point 2 above, that also afterwards validates the instance with the function from point 1).
4. a data structure J for problem instances holding all the instance information, with a function that can validate whether the instance has consistent information and that throws an exception otherwise
5. a way to write an object of J to a text file (that also first validates the instance with the function from point 4)
6. a way to read an object of J from  a text file (compatible with point 5 above, that also afterwards validates the object with the function from point 4)
7. an function implementation of the objective function F that uses the information from an object of J, takes as input an object of X, and returns the objective value
