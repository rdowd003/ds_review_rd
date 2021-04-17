## Converting string to int from binary ==> int(str,2)

def binary_array_to_number(arr):
    return int(''.join(str(a) for a in arr), 2)