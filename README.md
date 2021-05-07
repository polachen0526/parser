# parser
parser for node information
// -------------------- func_list --------------------------- ------- \\\n
read_json ::讀取excel和parser.json，excel是合併目標，而parser.json是模塊信息\n
split_layer_info_to_vector  :: spilt the  layer.information into vector and reutrn to "layer_info_data_vector"\n
layer_info_data_vector_trace:: trace every node ,if the node have the "class_name" like Conv2D,MaxPooling2D,Dropout..........,and give node Attributes\n
