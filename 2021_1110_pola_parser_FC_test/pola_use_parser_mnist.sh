g++ new_train.cpp -o new_train_2 -std=c++17 -fsanitize=address
./new_train_2 model_mnist.json model_mnist.csv FC_WEIGHT
