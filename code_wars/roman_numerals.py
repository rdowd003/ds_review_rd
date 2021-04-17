## INSTRUCTIONS
'''
Create a function that takes a Roman numeral as its argument 
and returns its value as a numeric decimal integer. You don't need 
to validate the form of the Roman numeral.

Modern Roman numerals are written by expressing each decimal digit 
of the number to be encoded separately, starting with the leftmost 
digit and skipping any 0s. So 1990 is rendered "MCMXC" 
(1000 = M, 900 = CM, 90 = XC) and 2008 is rendered "MMVIII" (2000 = MM, 8 = VIII). 
The Roman numeral for 1666, "MDCLXVI", uses each letter in descending order.

Example:
'''

##DID NOT SOLVE - got close, couldn't figure it out

## BEST PRACTICE

# Got this part. 
def solution(roman):
    dict = {
        "M": 1000,
        "D": 500,
        "C": 100,
        "L": 50,
        "X": 10,
        "V": 5,
        "I": 1
    }

# Invert the list - I alost did this but didn't try it. silly. 
    last, total = 0, 0
    for c in list(roman)[::-1]:
        if last == 0:
            total += dict[c]
        elif last > dict[c]:
            total -= dict[c]
        else:
            total += dict[c]
        last = dict[c]
    return total