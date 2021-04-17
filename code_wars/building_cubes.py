## INSTRUCTIONS:

'''


Your task is to construct a building which will be a pile of n cubes. 
The cube at the bottom will have a volume of n^3, the cube above will have 
volume of (n-1)^3 and so on until the top which will have a volume of 1^3.

You are given the total volume m of the building. Being given m can you 
find the number n of cubes you will have to build?

The parameter of the function findNb (find_nb, find-nb, findNb) will be 
an integer m and you have to return the integer n such as 
n^3 + (n-1)^3 + ... + 1^3 = m if such a n exists or -1 if there is no such n.


'''



## SOLUTION

def find_nb(m):
    _sum = 0
    n = 1
    s = m**(1/2)
    
    if (round(m**(1/2)))**2 == m:
        while(_sum < s): 
            _sum += n 
            n+=1
        n-=1
        if _sum == s: 
            return n 
        else:
            return -1
    else:
        return -1


## BEST PRACTICE:

def find_nb(m):
    n = 1
    volume = 0
    while volume < m:
        volume += n**3
        if volume == m:
            return n
        n += 1
    return -1


## NOTES:
# 
#  sum of cubes: 
# (n(n+1)/2)^2 = sum, which also = sum of first n integers

   