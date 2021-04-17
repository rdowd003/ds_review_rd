## INSTRUCTIONS

'''
Implement the function unique_in_order which takes as 
argument a sequence and returns a list of items without 
any elements with the same value next to each other and 
preserving the original order of elements.

unique_in_order('AAAABBBCCDAABBB')      == ['A', 'B', 'C', 'D', 'A', 'B']
unique_in_order('ABBCcAD')              == ['A', 'B', 'C', 'c', 'A', 'D']
unique_in_order([1,2,2,3,3])            == [1,2,3]


'''

## SOLUTION

def unique_in_order(iterable):
    if iterable:
        lst1 = list(iterable)
        lst2 = []
        lst2.append(lst1[0])
        
        for val in lst1[1:]: 
            if val != lst2[-1]: 
                lst2.append(val) 
        
        return lst2
    else:
        return []


## BEST PRACTICE

def unique_in_order(iterable):
    result = []
    prev = None                     # this good - fixes beginning edge case
    for char in iterable[0:]:
        if char != prev:
            result.append(char)
            prev = char
    return result



## CLEVER

from itertools import groupby

def unique_in_order(iterable):
    return [k for (k, _) in groupby(iterable)]