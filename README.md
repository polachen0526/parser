# parser
parser for node information
//--------------------func_list----------------------------------\\
read_json                   :: read excel and parser.json,the excel is the merge target and the parser.json is module information
split_layer_info_to_vector  :: spilt the  layer.information into vector and reutrn to "layer_info_data_vector"
layer_info_data_vector_trace:: trace every node ,if the node have the "class_name" like Conv2D,MaxPooling2D,Dropout..........,and give node Attributes
