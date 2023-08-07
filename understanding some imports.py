import numpy as np
my_2d_array = np.array([[1,2,3,4], [5,6,7,8]], dtype=np.int64)

# Print out memory address
print(my_2d_array.data)

# Print out the shape of `my_array`
print(my_2d_array.shape)

# Print out the data type of `my_array`
print(my_2d_array.dtype)

# Print out the stride of `my_array`
print(my_2d_array.strides)