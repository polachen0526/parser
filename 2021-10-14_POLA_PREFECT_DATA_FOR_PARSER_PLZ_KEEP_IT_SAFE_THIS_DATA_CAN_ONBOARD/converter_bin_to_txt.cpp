#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>
#include <cstring>
#include <string>
#include <cmath>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#define FMT_HEADER_ONLY
#include <fmt/format.h>
struct layer_info{
    uint8_t data[2];
};
off_t fsize(const char *filename) {
    struct stat st; 

    if (stat(filename, &st) == 0)
        return st.st_size;

    return -1; 
}

std::vector<layer_info> parse_file(const std::string filename){
    int fd = open(filename.c_str(), O_CREAT | O_RDWR | O_SYNC, S_IRUSR | S_IWUSR);
    if(fd < 0)
        throw std::invalid_argument("Can not open layer bin file.");
    
    std::vector<layer_info> tmp;
    int file_size = (int) fsize(filename.c_str());

    //if(file_size%20 != 0){
    //    throw "Layer bin file format error.";
    //}
    std::cout<<file_size<<std::endl;
    tmp.resize(file_size/2);

    void* map_memory = mmap(0, file_size, PROT_READ, MAP_SHARED, fd, 0);
    memcpy(tmp[0].data, map_memory, file_size);
    std::cout<<tmp[0].data<<std::endl;
    munmap(map_memory, file_size);
    uint16_t * tmp_test = (uint16_t*)tmp[0].data;//2010224
    for(int x = 2006720; x < 2006730; x++){
        //std::cout << std::hex << fmt::format("{:08x} ",tmp_test[x+4]);
        //std::cout << std::hex << fmt::format("{:08x} ",tmp_test[x+3]);
        //std::cout << std::hex << fmt::format("{:08x} ",tmp_test[x+2]);
        //std::cout << std::hex << fmt::format("{:08x} ",tmp_test[x+1]);
        std::cout << x <<" : "<<std::hex << fmt::format("{:08x} ",tmp_test[x]);
        std::cout<<std::endl;
    }
    close(fd);
    
    return tmp;
}

int main(int argc , char **argv){
    const std::string layer_info_file = argv[1];
    const char* layer_info_file_c = layer_info_file.c_str();
    auto tmp_vec = parse_file(layer_info_file_c);
    //std::cout<<tmp_vec.size()<<std::endl;
    //for(auto i : tmp_vec){
    //    std::cout<<i.data<<std::endl;
    //}
}