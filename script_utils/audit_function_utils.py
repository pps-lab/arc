import Compiler.types as types
import Compiler.library as library


def compute_median(X_arr):
    """Compute the median of an Array.
    
    This function computes the median of an array by sorting a copy of the provided array and either returns the arithmetic mean of the
    middle elements of the sorted array (for an array of even length), or returns the middle element of the sorted array (for an array of 
    odd length). The function expects that X_arr is an instance of Array() 
    """
    X_copy = types.Array(X_arr.length, X_arr.value_type)
    X_copy.assign(X_arr)
    X_copy.sort()
    the_median = 0
    if X_arr.length % 2 == 0:
        the_median = (X_copy[(X_copy.length-1)//2] + X_copy[X_copy.length//2])/2
    else:
        the_median = X_copy[X_copy.length//2]
    return the_median

def compute_abs_diff_value(X_arr, the_val):
    """This functions computes the absolute difference of each array entry of X_arr and the_val.

    The function expects X_arr to be of type Array and to be not None.
    It also expects the_val to be not None and an instance of a compatible value type of the value type of X_arr 
    """
    X_copy = types.Array(X_arr.length, X_arr.value_type)
    @library.for_range(start=0,stop=X_arr.length,step=1)
    def _(i):
        X_copy[i] = abs(X_arr[i] - the_val)
    return X_copy

# We expect an Array
def MAD_Score(X_array):
    """This function computes the MAD_Score of X_array.
    
    The function expects X_array to be not None and to be an instance of Array.
    The MAD_Score is defined as: MAD_Score(X_arr) = (X_arr - median(X_Arr))/(median_absolute_difference(X_arr))
    """
    score_array = types.Array(X_array.length, X_array.value_type)
    X_median = compute_median(X_array)
    X_abs_diff = compute_abs_diff_value(X_array, X_median)
    X_abs_median = compute_median(X_abs_diff)
    
    @library.for_range(start=0,stop=X_array.length,step=1)
    def _(i):
        score_array[i] = (X_array[i] - X_median)/(X_abs_median)
    return score_array

    


        
