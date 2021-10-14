g++ new_train.cpp -o new_train_2 -std=c++17 -fsanitize=address
./new_train_2 model_yolo256_new.json model_yolo256_design.csv WEIGHT
