import numpy as np
'''
activate func exteneded code 2022/02/27
use this func to close the value instand of hard_sigmoid
'''
def shift_result(value,shift_number):
    return value*2**shift_number

def fixed_sigmoid_lookuptable(input_value,shift_number):
    #data range
    first_value = shift_result(5,shift_number)
    second_value = shift_result(2.375,shift_number) #2**1+2**-2+2**-3
    third_value = shift_result(1,shift_number)
    abs_value = shift_result(abs(input_value),shift_number)

    if(abs_value >= first_value):
        return shift_result(1,shift_number)
    elif(abs_value >= second_value and abs_value < first_value):
        return shift_result(abs_value,-5) + shift_result(0.84375,shift_number) #2**-1+2**-2+2**-4+2**-5
    elif(abs_value >= third_value and abs_value < second_value):
        return shift_result(abs_value,-3) + shift_result(0.625,shift_number)#2**-1+2**-3
    elif(abs_value >= 0 and abs_value < third_value):
        return shift_result(abs_value,-2) + shift_result(0.5,shift_number)#2**-1
    else:
        print("please check your input number !!!!")

def nsigmoid(input_value):
    z = np.exp(-input_value)
    sig = 1/(1+z)
    return sig
def ntanh(input_value):
    z = np.exp(-2*input_value)
    tan = 2/(1+z)-1
    return tan
def check_answer(value):
    A_a = nsigmoid(value)
    A_b = fixed_sigmoid_lookuptable(value,10) / shift_result(1,10)
    if(value>=0):
        print("your value is positive")
        print("the A_a result is {}".format(A_a))
        print("the A_b result is {}".format(A_b))
        print("the tan A_a result is {}".format(ntanh(value)))
        print("the tan A_b result is {}".format(2*fixed_sigmoid_lookuptable(2*value,10) / shift_result(1,10)-1))
    else:
        print("your value is negative")
        print("the A_a result is {}".format(A_a))
        print("the 1-A_b result is {}".format(1-A_b))
        print("the tan A_a result is {}".format(ntanh(value)))
        print("the tan A_b result is {}".format(2*(1-(fixed_sigmoid_lookuptable(2*value,10) / shift_result(1,10)))-1))
        A_b = 1 - A_b
    return A_a-A_b
