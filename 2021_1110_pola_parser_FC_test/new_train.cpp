#include <iostream>
#include <cstring>
#include <sstream>
#include <vector>
#include <map>
#include <fstream>
#include <iomanip>
#include <nlohmann/json.hpp>
#include <memory>
#include <algorithm>
#include <bitset>
#include <string>
#include <filesystem>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>    
#include <unistd.h>
#define FMT_HEADER_ONLY
#include <fmt/format.h>
#include <cassert>
using json = nlohmann::json;
//enum File_Type : size_t {Weight, Bias, Scale, Var, Mean};
#define DATA_Depth_24       24
#define MAX_TILE_SIZE   32 //34
#define BUS_WIDTH       16        //AXI BUS 16-Byte
#define DATA_WIDTH       2        //Data 16-bit
#define DATA_DEPTH     (BUS_WIDTH/DATA_WIDTH)
#define MAX_ICH        (BUS_WIDTH/DATA_WIDTH)
#define DUMP_WEIGHT
#define DUMP_LAYER_INFO
#define DUMP_TILE_INFO
//#define DUMP_FOR_SIM
#define HW_INFO_MAX 1024
#define weight_len_not_last      72 //說明這邊要考慮是否是在最後一個input_channel如果是你需要給予74 不是的話給72，這邊和AXI有關
#define weight_len_last_3        74
#define weight_len_last_1        78
#define ONCE_CAL_WEIGHT_NUM_3 (DATA_DEPTH*DATA_DEPTH*9)             //(DATA_DEPTH*DATA_DEPTH*9 + DATA_DEPTH*2)
#define ONCE_CAL_WEIGHT_NUM_1 (DATA_Depth_24* DATA_Depth_24 *1)    //(DATA_Depth_24* DATA_Depth_24 *1) + (DATA_Depth_24 *2))

#ifdef DUMP_FOR_SIM
    //#define DATA_BASE_ADDR    0x01002000 
    //#define WEIGHT_BASE_ADDR  0x02001000
    //#define LAYER_ADDR_OFFSET 0x10000
    #define DATA_BASE_ADDR    0x6fdb0000 
    #define WEIGHT_BASE_ADDR  0x6fd05000
    #define LAYER_ADDR_OFFSET 0x10000
#else
    #define DATA_BASE_ADDR    0
    #define WEIGHT_BASE_ADDR  0
    #define LAYER_ADDR_OFFSET 0
#endif
template<typename T>
size_t size_in_byte(const std::vector<T> &a){
    return a.size()*sizeof(T);
}
struct File_Type_define{
    size_t weight;
    size_t bias;
    size_t beta;
    size_t gamma;
    size_t variance;
    size_t mean;
} File_Attributes;
class pstruct{
    public:
        json layer_info;
        json layer_target;
        json layer_quant;
        json output_layers;
        std::vector<json> layer_info_data_vector;
        std::vector<std::shared_ptr<pstruct>>layer_info_data_pointer_vector;
        std::vector<std::vector<int>>target_layer_info_vector;
        std::vector<std::vector<int>>target_layer_quant_vector;
        std::vector<std::shared_ptr<pstruct>>merge_node_vector;
        std::vector<size_t>out_addr_set;
        std::vector<size_t>out_addr_set_pool;
        std::vector<u_char>tile_padding_type;
        size_t  Tile_Info_Number = 0;
        std::vector<uint32_t> tile_info;
        //----------------------func---------------------------
        std::tuple<json,json,json,json>read_json(const std::string filename,const std::string info_design);
        void scan_table(const json layers,const std::string output_layer); //TODO for concat you need take out the address and concat_control
        std::vector<json>split_layer_info_to_vector(const json &data_info);
        std::vector<std::shared_ptr<pstruct>>layer_info_data_vector_trace(const std::vector<json>&layer_info_data_vector);\
        void weight_offset(std::vector<std::shared_ptr<pstruct>> &layer_info_data_pointer_vector,std::vector<std::vector<int>> &target_layer_info_vector);
        std::vector<std::vector<int>>target_node_trace(json &target_info);
        std::vector<std::vector<int>>target_quant_trace(json &target_quant);
        std::vector<std::shared_ptr<pstruct>> merge_node(std::vector<std::shared_ptr<pstruct>> &layer_info_data_pointer_vector,std::vector<std::vector<int>> &target_layer_info_vector,std::vector<std::vector<int>> &target_layer_quant_vector,std::vector<int>&merge_node_jump_location);
        void merge_node_fix(std::vector<std::shared_ptr<pstruct>> &merge_node_vector,std::vector<int>&merge_node_jump_location,std::vector<int>&branch_node_jump_location);
        void get_tile_info(std::vector<std::shared_ptr<pstruct>> &merge_node_vector);
        void gen_out_addr(std::shared_ptr<pstruct> &merge_node);
        void gen_t_type_m(std::shared_ptr<pstruct> &merge_node);
        std::vector<uint32_t>Gen_Layer_Info(std::shared_ptr<pstruct> &merge_node);
        void dump_total_tile_sim(const std::string &filename,const std::vector<std::shared_ptr<pstruct>> &merge_node_vector);
        std::vector<uint32_t>gen_layer_info_data(std::vector<std::shared_ptr<pstruct>>&merge_node_vector);
        void dump_layer_info_sim(const std::string &filename,const std::vector<uint32_t> &layer_info_data);
        void dump_layer_info_data_bin(const std::string &filename, const std::vector<uint32_t> &layer_info_data);
        //std::vector<short>gen_total_weight(const std::string &dir_path,const std::vector<std::shared_ptr<pstruct>> &merge_node_vector);
        std::vector<std::vector<short>>gen_total_weight(const std::string &dir_path,const std::vector<std::shared_ptr<pstruct>> &merge_node_vector);
        std::vector<std::string>gen_weight_path(const std::string &dir_path,const std::shared_ptr<pstruct> &merge_node);
        std::vector<short>gen_layer_weight(const std::shared_ptr<pstruct> &merge_node,const std::vector<std::string> &path);
        void dump_weight_sim(const std::string &filename,const std::vector<short>&weight_data);
        void dump_tile_bin(const std::string &filename, const std::vector<std::shared_ptr<pstruct>> &merge_node_vector);
        void dump_weight_bin(const std::string &filename, const void *src, const size_t src_size);
        void fc_divd ( int32_t node, int32_t &dimx, int32_t &dimy, int32_t &dimz);
        //----------------------parameter----------------
        int32_t input_feature_size   = -1;
        int32_t input_channel        = -1;
        int32_t output_channel       = -1;
        int32_t output_feature_size  = -1;
        int32_t kernel_size          = -1;
        int32_t kernel_stride        = -1;
        int32_t pool_size            = -1;
        int32_t pool_stride          = -1;
        int32_t weight_address       = -1;
        int32_t tmp                  = -1;
        int32_t units                = -1;
        int32_t n_none               = -1;
        int32_t input_padding_size   =  0; //always be zero to prevent the class_name is not conv2D
        int32_t output_padding_size  = -1;
        int32_t input_tile_size      = -1;
        int32_t input_tile_number    = -1;
        int32_t output_tile_size     = -1;
        int32_t output_tile_number   = -1;
        int32_t next_tile_size       = -1;
        int32_t branch_input_tile_size   = -1; //現在看他和Input_tile_size不是一樣ㄇ，但是如果你padding那邊不是same的話，如果今天是valid那你這邊就會和input_tile_size差看看幾個CONV，因為你沒有PADDING，這邊是從output_feature_size回推
        int32_t output_address       = -1;
        int32_t pool_address         = 0;
        int32_t Bit_Serial           = 0;
        int32_t quant_batch_bias     = 0;
        int32_t quant_finish         = 0;
        int32_t pooling_quant_finish = 0;
        int32_t quant_batch          = 0;
        int32_t quant_word_size      = 0;
        int32_t quant_obuf           = 0;
        int32_t Tile_Info_Addr       = 0;
        int32_t Upsample_size        = 0;
        int32_t Dense_size_input_x   = 0; //for Dense_size_input_x
        int32_t Dense_size_input_y   = 0; //for Dense_size_input_y
        int32_t Dense_size_output_x  = 0; //for Dense_size_output_x
        int32_t Dense_size_output_y  = 0; //for Dense_size_output_y
        int32_t Dense_input_node     = 0;
        int32_t Dense_output_node    = 0;
        int32_t Dense_input_channel  = 0;
        int32_t Dense_output_channel = 0;
        std::vector<int32_t>valid_address_vec;
        size_t Leaky_ReLU_alpha_FP = 0; //Fixed point
        std::vector<int32_t> input_address;
        std::vector<int32_t> Previous_node_OCH;
        std::string weight_name      = "none";
        std::string class_name       = "none";
        std::string padding          = "none";
        std::string activate         = "none";
        std::string pool_padding     = "none";
        std::string s_none           = "none";
        std::string node_name        = "none";
        std::vector<std::string>node_name_vector;
        float Leaky_ReLU_alpha       = -1;
        bool Batch_First             = false;
        bool Is_LeakyReLU            = false;
        bool Have_bias               = false;
        bool Have_BatchNormalization = false;
        bool Have_ReLU               = false;
        bool Have_Flatten            = false;
        bool Have_Dense              = false;
        bool Have_Maxpooling         = false;
        bool Have_Upsample           = false;
        bool Have_Concat             = false;
        bool Concat_output_control   = false;
        bool branch_node             = false;
        bool concat_node             = false;
        bool IF_PRE_NODE_IS_DENSE    = false;
        //find concat parameter
        std::vector<int32_t> branch_node_location;
        std::vector<int>merge_node_jump_location;
        std::vector<int>branch_node_jump_location;
        std::map<std::string,bool> scan_result_map;//哪個點被走過兩次
        std::map<std::string,bool> scan_result_branch_node_map;//抓出下一個分支點
        std::map<std::string,int>concat_map_vector;//各種不同的concat節點裝到一個vector
        void algorithm_for_basic_conv(const std::shared_ptr<pstruct> &merge_node,const size_t &Depth_select,const size_t &once_cal_weight_num_select,const size_t &times_calc,const size_t &OCH_NUM,const size_t &ICH_NUM, const size_t &HW_KERNEL_SIZE,const std::vector<std::vector<float>> &data_in,std::vector<short> &weight_data);
        void algorithm_for_Dense(const std::shared_ptr<pstruct> &merge_node,const size_t &Depth_select,const size_t &once_cal_weight_num_select,const size_t &times_calc,const size_t &OCH_NUM,const size_t &ICH_NUM, const size_t &HW_KERNEL_SIZE,const std::vector<std::vector<float>> &data_in,std::vector<short> &weight_data);
        
    private:
        size_t round_ch_8(const size_t ch);
        size_t round_ch_24(const size_t ch);
        inline size_t calc_input_tile_number(const std::shared_ptr<pstruct> &pinfo);
        inline size_t calc_output_tile_number(const std::shared_ptr<pstruct> &pinfo);
        inline size_t calc_layer_tile_size(const std::shared_ptr<pstruct> &pinfo);
        inline size_t calc_next_layer_tile_size(const std::shared_ptr<pstruct> &pinfo);
        inline size_t calc_branch_next_layer_tile_size(const std::shared_ptr<pstruct> &pinfo);
        inline size_t calc_output_layer_tile_size(const std::shared_ptr<pstruct> &pinfo);
        inline size_t calc_dense_input_tile_number(const std::shared_ptr<pstruct> &pinfo);
        inline size_t calc_dense_output_tile_number(const std::shared_ptr<pstruct> &pinfo);
        inline size_t calc_dense_layer_tile_size(const std::shared_ptr<pstruct> &pinfo);
        inline size_t calc_dense_next_layer_tile_size(const std::shared_ptr<pstruct> &pinfo);
        inline size_t calc_dense_branch_next_layer_tile_size(const std::shared_ptr<pstruct> &pinfo);
        inline size_t calc_dense_output_layer_tile_size(const std::shared_ptr<pstruct> &pinfo);
};
size_t pstruct::round_ch_8(const size_t ch){
    return std::ceil(ch/(double)DATA_DEPTH);
}
size_t pstruct::round_ch_24(const size_t ch){
    return std::ceil(ch/(double)DATA_Depth_24);
}
//----------------------FOR CONV SERIES-----------------------------
inline size_t pstruct::calc_input_tile_number(const std::shared_ptr<pstruct> &pinfo){
    const size_t tile_size = MAX_TILE_SIZE;//- pinfo->input_padding_size*2;
    const size_t tmp = std::ceil(pinfo->input_feature_size/(double)tile_size);
    return (tmp >0) ? tmp : 1;
}
inline size_t pstruct::calc_output_tile_number(const std::shared_ptr<pstruct> &pinfo){
    const size_t tile_size = MAX_TILE_SIZE;//- pinfo->output_padding_size*2;
    const size_t tmp = std::ceil(pinfo->output_feature_size/(double)tile_size);
    return (tmp > 0) ? tmp : 1;
}
inline size_t pstruct::calc_layer_tile_size(const std::shared_ptr<pstruct> &pinfo){
    const size_t tmp = pinfo->input_feature_size;// + pinfo->input_padding_size*2;
    return (MAX_TILE_SIZE > tmp) ? tmp : MAX_TILE_SIZE;
}
inline size_t pstruct::calc_next_layer_tile_size(const std::shared_ptr<pstruct> &pinfo){
    const size_t tmp = pinfo->output_feature_size;// + pinfo->output_padding_size*2;
    return (MAX_TILE_SIZE > tmp) ? tmp : MAX_TILE_SIZE;
}
inline size_t pstruct::calc_branch_next_layer_tile_size(const std::shared_ptr<pstruct> &pinfo){
    const size_t tmp = (pinfo->Have_Maxpooling) ? (pinfo->output_feature_size * pinfo->pool_size) : (pinfo->output_feature_size);
    return (MAX_TILE_SIZE > tmp) ? tmp : MAX_TILE_SIZE;
}
inline size_t pstruct::calc_output_layer_tile_size(const std::shared_ptr<pstruct> &pinfo){
    //size_t tmp = (pinfo->input_tile_size - (pinfo->input_padding_size*2))/((pinfo->kernel_stride!=0) ? pinfo->kernel_stride : 1);
    size_t tmp = pinfo->input_tile_size /  (pinfo->kernel_stride!=0 ? pinfo->kernel_stride : 1); 
    if(pinfo->pool_stride>0)
        tmp = tmp/pinfo->pool_stride;
    return (MAX_TILE_SIZE > tmp) ? tmp : MAX_TILE_SIZE;
}
//----------------------FOR FULLY CONNECT(dense)-----------------------------
inline size_t pstruct::calc_dense_input_tile_number(const std::shared_ptr<pstruct> &pinfo){
    size_t tmp;
    if(pinfo->IF_PRE_NODE_IS_DENSE)
        tmp = pinfo->Dense_input_node;//由前一層的dense f_och * f_ox * f_oy組成
    else
        tmp = std::pow(pinfo->input_feature_size,2) * round_ch_24(pinfo->input_channel)*DATA_Depth_24;
    return (tmp >0) ? tmp : 1;
}
inline size_t pstruct::calc_dense_output_tile_number(const std::shared_ptr<pstruct> &pinfo){
    const size_t tmp = pinfo->Dense_output_channel * pinfo->Dense_size_output_x * pinfo->Dense_size_output_y;
    return (tmp > 0) ? tmp : 1;
}
inline size_t pstruct::calc_dense_layer_tile_size(const std::shared_ptr<pstruct> &pinfo){
    const size_t tmp = 1;
    return (MAX_TILE_SIZE > tmp) ? tmp : MAX_TILE_SIZE;
}
inline size_t pstruct::calc_dense_next_layer_tile_size(const std::shared_ptr<pstruct> &pinfo){
    const size_t tmp = 1;
    return (MAX_TILE_SIZE > tmp) ? tmp : MAX_TILE_SIZE;
}
inline size_t pstruct::calc_dense_branch_next_layer_tile_size(const std::shared_ptr<pstruct> &pinfo){
    const size_t tmp = 1;
    return (MAX_TILE_SIZE > tmp) ? tmp : MAX_TILE_SIZE;
}
inline size_t pstruct::calc_dense_output_layer_tile_size(const std::shared_ptr<pstruct> &pinfo){
    const size_t tmp = 1;
    return (MAX_TILE_SIZE > tmp) ? tmp : MAX_TILE_SIZE;
}
//-----------------------------------------------------------------------
void pstruct::algorithm_for_Dense(const std::shared_ptr<pstruct> &merge_node,const size_t &Depth_select,const size_t &once_cal_weight_num_select,const size_t &times_calc,const size_t &OCH_NUM,const size_t &ICH_NUM, const size_t &HW_KERNEL_SIZE,const std::vector<std::vector<float>> &data_in,std::vector<short> &weight_data){
    std::ofstream ofs;
    ofs.open(merge_node->node_name+".txt");
    size_t kernel_count = 0;
    std::cout<<"OCH_NUM!!!!!!!!!!! "<<OCH_NUM<<std::endl;
    for(size_t iy_index = 0; iy_index < merge_node->Dense_size_input_y;iy_index++){
        for(size_t ix_index =0;ix_index < merge_node->Dense_size_input_x;ix_index++){
            for(size_t ich_count = 0;ich_count<ICH_NUM;ich_count++){
                for(size_t ich = 0;ich<Depth_select;ich++){
                    for(size_t oy_index=0;oy_index<merge_node->Dense_size_output_y;oy_index++){
                        for(size_t ox_index=0;ox_index<merge_node->Dense_size_output_x;ox_index++){
                            for(size_t och_count=0;och_count<OCH_NUM;och_count++){
                                for(size_t three_rd_out=0;three_rd_out<times_calc;three_rd_out++){
                                    for(size_t och=0;och<DATA_DEPTH;och++){
                                        short t=0;
                                        size_t index;
                                        size_t och_index, o_index;
                                        size_t ich_index, i_index;
                                        och_index = ( och + three_rd_out * DATA_DEPTH + och_count * Depth_select );
                                        o_index = 
                                                ( oy_index  * Depth_select  *   merge_node->Dense_size_output_x
                                                + ox_index  * Depth_select
                                                + och_index );

                                        ich_index = ( ich_count  * Depth_select
                                                    + ich );
                                        i_index = ( !merge_node->IF_PRE_NODE_IS_DENSE ) ? //軟體出發點1568 32*7*7
                                                ( ich_count * Depth_select //* merge_node->Dense_size_input_x * merge_node->Dense_size_input_y
                                                + iy_index  * merge_node->Dense_input_channel * merge_node->Dense_size_input_x
                                                + ix_index  * merge_node->Dense_input_channel      
                                                + ich )
                                              : ( iy_index  * Depth_select  *   merge_node->Dense_size_input_x
                                                + ix_index  * Depth_select      
                                                + ich );    

                                        index = ich_count   *   once_cal_weight_num_select * merge_node->Dense_size_output_x * merge_node->Dense_size_output_y * merge_node->Dense_size_input_x * merge_node->Dense_size_input_y 
                                              + iy_index    *   once_cal_weight_num_select * merge_node->Dense_size_output_x * merge_node->Dense_size_output_y * merge_node->Dense_size_input_x
                                              + ix_index    *   once_cal_weight_num_select * merge_node->Dense_size_output_x * merge_node->Dense_size_output_y 
                                              + ich
                                              + oy_index    *   once_cal_weight_num_select * merge_node->Dense_size_output_x
                                              + ox_index    *   once_cal_weight_num_select
                                              + three_rd_out*   Depth_select
                                              + och         *   Depth_select               * 3;

                                        index += ( ich_count == ICH_NUM-1 && iy_index == merge_node->Dense_size_input_y-1 && ix_index == merge_node->Dense_size_input_x-1 ) ? (Depth_select * 2 * (ox_index +oy_index *merge_node->Dense_size_output_x) ) : 0 ;
                                        //                            256                                         160                                             16                                              24
                                        if ( i_index < merge_node->Dense_input_node && o_index < merge_node->Dense_output_node && ich_index < merge_node->Dense_input_channel && och_index < merge_node->Dense_output_channel ){
                                            t = data_in[File_Attributes.weight][kernel_count] * std::pow(2,merge_node->quant_batch);
                                            kernel_count ++;
                                        }
                                        //printf("%d , %d , %d , %d , %d , %d , %d , %d , %d , %d\n",iy_index,ix_index,ich_count,ich,oy_index,ox_index,och_count,three_rd_out,och,kernel_count);
                                        ofs<<" iy_index "<<iy_index<<" ix_index "<<ix_index<<" ich_count "<<ich_count<<" ich "<<ich<<" oy_index "<<oy_index<<" ox_index "<<ox_index<<" och_count "<<och_count<<" three_rd_out "<<three_rd_out<<" och "<<och<<" kernel_count "<<kernel_count<<" index "<<index<<"\n";
                                        weight_data[index] = t;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    kernel_count = 0;
    for(size_t alpha_beta=0;alpha_beta<2;alpha_beta++){
        for(size_t oy_index=0;oy_index<merge_node->Dense_size_output_y;oy_index++){
            for(size_t ox_index=0;ox_index<merge_node->Dense_size_output_x;ox_index++){
                for(size_t och_count=0;och_count<OCH_NUM;och_count++){
                    for(size_t three_rd_out=0;three_rd_out<times_calc;three_rd_out++){
                        for(size_t och=0;och<DATA_DEPTH;och++){
                            size_t och_index, o_index , index;
                            short t = 0;
                            och_index = ( och + three_rd_out * DATA_DEPTH + och_count * Depth_select );
                            o_index = 
                                    ( oy_index  * Depth_select  *   merge_node->Dense_size_output_x
                                    + ox_index  * Depth_select
                                    + och_index );
                                        //   " 準 備 走 Input feature map 最 後 一 個 點  ------------------------------------------------------------"  " Output Feauture Map 點總數 -----------------------------------------------------------" 這只是有開始擺放Normalizaiton的offset
                            index   = ( ( merge_node->Dense_size_input_y * merge_node->Dense_size_input_x * ICH_NUM -1 ) * Depth_select ) * ( merge_node->Dense_size_output_y * merge_node->Dense_size_output_x * 1 * Depth_select );
                            index   += once_cal_weight_num_select;
                            index   += oy_index * ( once_cal_weight_num_select + Depth_select*2) * merge_node->Dense_size_output_x;
                            index   += ox_index * ( once_cal_weight_num_select + Depth_select*2);
                            index   += three_rd_out * DATA_DEPTH;
                            index   += och;
                            index   += alpha_beta * Depth_select;
                            if ( alpha_beta == 0 ){ // This is 1 
                                t = 1 * std::pow(2,merge_node->quant_batch);
                                weight_data[index] = t;
                            } else if ( alpha_beta == 1 && o_index < merge_node->Dense_output_node && och_index < merge_node->Dense_output_channel ) { 
                                t = data_in[File_Attributes.bias][kernel_count] * std::pow(2,merge_node->quant_batch);
                                weight_data[index] = t;
                                kernel_count ++;
                            }
                            ofs<<" oy_index "<<oy_index<<" ox_index "<<ox_index<<" och_count "<<och_count<<" three_rd_out "<<three_rd_out<<" och "<<och<<" kernel_count "<<kernel_count<<" index "<<index<<"\n";


    }}}}}}
    ofs.close();
}
void pstruct::algorithm_for_basic_conv(const std::shared_ptr<pstruct> &merge_node,const size_t &Depth_select,const size_t &once_cal_weight_num_select,const size_t &times_calc,const size_t &OCH_NUM,const size_t &ICH_NUM, const size_t &HW_KERNEL_SIZE,const std::vector<std::vector<float>> &data_in,std::vector<short> &weight_data){
    std::ofstream ofs;
    ofs.open(merge_node->node_name+".txt");
    size_t kernel_count = 0;
    for(size_t och_count = 0;och_count<OCH_NUM;och_count++){//為了要一次輸出可以連續，你必須把東西猜開放到不同的bank，才能把資料連續傳送，舉例來說，如果你要傳0-7但是你又放在同一個bank那你沒辦法一次傳，所以正常連續擺放會變成一個bank0-8-15-23-...第二個1-9-16-24-...第三個2-10-17-25-...這樣才可以讓AXI拿到連續的資料雖然看起來很怪，但是每個bank確實可以一次丟一筆資料出去
        size_t index;
        for(size_t three_rd_out = 0; three_rd_out<times_calc;three_rd_out++){
            for(size_t och = 0;och<DATA_DEPTH;och++){//0-8-15他這邊要考慮到bank的問題，所以要把資料倒著放
                for(size_t ich_count = 0; ich_count<ICH_NUM; ich_count++){
                    for(size_t three_rd_in = 0; three_rd_in<times_calc;three_rd_in++){
                        for(size_t ich=0;ich<DATA_DEPTH;ich++){
                            for(size_t k=0;k<HW_KERNEL_SIZE;k++){//TODO 2021/8/28 這邊的KERNEL_SIZE原本是設定9讓不管是1*1 OR 3*3都是用9去做
                                short t=0;
                                if(merge_node->kernel_size==1){
                                    if((och_count*Depth_select+och+(DATA_DEPTH*three_rd_out)) < merge_node->output_channel){ //這邊的data_depth 要改成 Depth_select 讓上面去判定你這邊應該會有多少資料 8or24
                                        if((ich_count*Depth_select+ich+(DATA_DEPTH*three_rd_in))<merge_node->input_channel){
                                            #ifdef BATCH_MERGE_WEIGHT
                                                if(merge_node->Have_BatchNormalization){
                                                    const size_t och_index = och_count*DATA_DEPTH*three_rd_out+och;
                                                    float weight_tmp = data_in[File_Attributes.weight][kernel_count]*(data_in[File_Attributes.gamma][och_index] / std::sqrt(data_in[File_Attributes.variance][och_index]));
                                                    t = weight_tmp * std::pow(2,merge_node->quant_batch);
                                                    if(weight_tmp * (short)t < 0){
                                                        std::cout << fmt::format("och_index : {}\n", och_index) << std::endl;
                                                        std::cout << fmt::format("Weight : {}, Scale : {}, After : {}, Fix : {}\n",data_in[File_Attributes.weight][kernel_count], 
                                                        (data_in[File_Attributes.gamma][och_index] / std::sqrt(data_in[File_Attributes.variance][och_index])), weight_tmp,(short)t);
                                                    }
                                                    assert(weight_tmp * (short)t >= 0);
                                                }else{
                                                    t = data_in[File_Attributes.weight][kernel_count] * std::pow(2,merge_node->quant_batch);
                                                    assert(data_in[File_Attributes.weight][kernel_count] * (short)t >= 0);
                                                }
                                            #else
                                                t = data_in[File_Attributes.weight][kernel_count] * std::pow(2,merge_node->quant_batch);
                                                assert(data_in[File_Attributes.weight][kernel_count] * (short)t >= 0);
                                            #endif
                                            if(k == HW_KERNEL_SIZE-1){
                                                kernel_count++;
                                            } 
                                        }
                                    }
                                }else{
                                    if((och_count*Depth_select+och)<merge_node->output_channel){
                                        if((ich_count*Depth_select+ich)<merge_node->input_channel){
                                            #ifdef BATCH_MERGE_WEIGHT
                                                if(merge_node->Have_BatchNormalization){
                                                    const size_t och_index = och_count*Depth_select+och;
                                                    float weight_tmp = data_in[File_Attributes.weight][kernel_count]*(data_in[File_Attributes.gamma][och_index] / std::sqrt(data_in[File_Attributes.variance][och_index]));
                                                    t = weight_tmp * std::pow(2,merge_node->quant_batch);
                                                    if(weight_tmp * (short)t < 0){
                                                        std::cout << fmt::format("och_index : {}\n", och_index) << std::endl;
                                                        std::cout << fmt::format("Weight : {}, Scale : {}, After : {}, Fix : {}\n",data_in[File_Attributes.weight][kernel_count], 
                                                        (data_in[File_Attributes.gamma][och_index] / std::sqrt(data_in[File_Attributes.variance][och_index])), weight_tmp, (short)t);
                                                    }
                                                    assert(weight_tmp * (short)t >= 0);
                                                }else{
                                                    t = data_in[File_Attributes.weight][kernel_count] * std::pow(2,merge_node->quant_batch);
                                                    assert(data_in[File_Attributes.weight][kernel_count] * (short)t >= 0);
                                                }
                                            #else
                                                t = data_in[File_Attributes.weight][kernel_count] * std::pow(2,merge_node->quant_batch);
                                                assert(data_in[File_Attributes.weight][kernel_count] * (short)t >= 0);
                                                //std::cout<<"weight "<<data_in[File_Attributes.weight][kernel_count] << " " << merge_node->quant_batch<<" "<<t<<" "<<kernel_count<<std::endl;
                                            #endif
                                            kernel_count++;
                                        }
                                    }
                                }
                                index = och_count       * once_cal_weight_num_select    * ICH_NUM + och_count   *   Depth_select    *   2
                                      + ich_count       * once_cal_weight_num_select
                                      + three_rd_out    * Depth_select                                            // In 1x1 : JUMP EVERY 8  ICH
                                      + och             * Depth_select                  * ( merge_node->kernel_size==1 ? 3 : 9 )                     // In 1x1 : JUMP EVERY 24 ICH
                                      + three_rd_in     * DATA_DEPTH                // In 1x1 : JUMP EVERY 8 ICH
                                      + k               * DATA_DEPTH                // In 3x3 : JUMP EVERY 8 ICH
                                      + ich;
                                weight_data[index] = t;
                                ofs<<index<<"\n";
                            }
                        }
                    }
                }
            }
        }
    }
    for(size_t i=0;i<OCH_NUM;i++){
        const size_t bias_scale_num = (merge_node->output_channel > (i+1)*Depth_select ) ? Depth_select : (merge_node->output_channel - i*Depth_select);
        for(size_t j=0;j<ICH_NUM;j++){//只要have_last_ich再放資料
            if(j==ICH_NUM-1){
                size_t index = (i*ICH_NUM+j) *  once_cal_weight_num_select + Depth_select * Depth_select * HW_KERNEL_SIZE; //前面的ONCE_CAL_WEIGHT_NUM意思是說妳已經做完第一次了所以理論上你的位置已經有(8*8*9+8*2)個，但是你後面加上的是說你的weight已經放好了，請你再把alpha bias放進去所以沒有8*2
                index  += i * Depth_select * 2;        
                for(size_t bias = 0; bias < Depth_select*2;bias++){
                    size_t och_index = bias%Depth_select + i*Depth_select;
                    short short_tmp = 0;
                    if(bias<Depth_select){
                        if(bias<bias_scale_num){
                            if(merge_node->Have_BatchNormalization){
                                #ifdef BATCH_MERGE_WEIGHT
                                    short_tmp = std::pow(2,merge_node->quant_batch);
                                #else
                                    short_tmp = (short)((data_in[File_Attributes.gamma][och_index] / std::sqrt(data_in[File_Attributes.variance][och_index])) * std::pow(2,merge_node->quant_batch));
                                    ofs<<"Alpha!!!!!! "<<"file_index:"<<och_index<<" Scale: "<<data_in[File_Attributes.gamma][och_index]<<" Var: "<<data_in[File_Attributes.variance][och_index]<<" quant_batch: "<<merge_node->quant_batch<<" ANS: "<<short_tmp<<"\n";
                                    if(data_in[File_Attributes.gamma][och_index] / std::sqrt(data_in[File_Attributes.variance][och_index]) * (short)short_tmp < 0){
                                        std::cerr << fmt::format("Scale {}, Var {}, Fix {}, Float {}\n", data_in[File_Attributes.gamma][och_index],
                                        std::sqrt(data_in[File_Attributes.variance][och_index]), (short)short_tmp, data_in[File_Attributes.gamma][och_index] / std::sqrt(data_in[File_Attributes.variance][och_index]));
                                    }
                                    assert(data_in[File_Attributes.gamma][och_index] / std::sqrt(data_in[File_Attributes.variance][och_index]) * short_tmp >= 0);
                                #endif
                            }else if(merge_node->Have_bias){
                                short_tmp = std::pow(2,merge_node->quant_batch);
                            }else{
                                short_tmp = 0;
                            }
                        }
                    }else{
                        if(bias<bias_scale_num+Depth_select){
                            if(merge_node->Have_BatchNormalization){
                                auto new_bias = data_in[File_Attributes.beta][och_index] - (data_in[File_Attributes.gamma][och_index] * data_in[File_Attributes.mean][och_index] / std::sqrt(data_in[File_Attributes.variance][och_index]));
                                short_tmp = (short)((new_bias) * std::pow(2,merge_node->quant_batch));
                                ofs<<"Beta!!!!!! "<<"file_index:"<<och_index<<" Bias: "<<data_in[File_Attributes.beta][och_index]<<" Scale: "<<data_in[File_Attributes.gamma][och_index]<<" Mean: "<<data_in[File_Attributes.mean][och_index]<<" Var: "<<data_in[File_Attributes.variance][och_index]<<" ANS: "<<short_tmp<<" quant_batch: "<<std::pow(2,merge_node->quant_batch)<<"\n";
                                assert((data_in[File_Attributes.beta][och_index] - (data_in[File_Attributes.gamma][och_index] * data_in[File_Attributes.mean][och_index] / std::sqrt(data_in[File_Attributes.variance][och_index]))) * (short)short_tmp >= 0);
                            }else if(merge_node->Have_bias){
                                short_tmp = (short)data_in[File_Attributes.beta][och_index] * std::pow(2,merge_node->quant_batch);
                            }else{
                                short_tmp = 0;
                            }
                        }
                    }
                    weight_data[index+bias] = short_tmp;
                    //ofs<<index<<"\n";
                }
            }
        }
    }
    ofs.close();
}

std::tuple<json,json,json,json>pstruct::read_json(const std::string filename,const std::string info_design){
    std::ifstream in(filename);
    json tmp = json::parse(in);
    json layers = tmp["config"]["layers"];
    json output_layers = tmp["config"]["output_layers"];
    //-------------------csv2json--------------------
    std::vector<std::map<std::string,std::vector<int>>>pola_target_vector;
    std::vector<std::map<std::string,std::vector<int>>>pola_quant_vector;
    std::fstream file;
    std::string line;
    file.open(info_design);
	while (getline( file, line,'\n'))
		{
            std::map<std::string,std::vector<int>>pola_target_map; 
            std::map<std::string,std::vector<int>>pola_quant_map; 
            std::vector<std::string>pola_data;
		    std::istringstream templine(line);
		    std::string data;
		    while (getline( templine, data,',')){
                pola_data.push_back(data);
            }
            if (pola_target_map.find(pola_data[0]) == pola_target_map.end() && !strncmp(pola_data[0].c_str(),"layers",5)){
                for(int i=1;i<pola_data.size();i++){
                    pola_target_map[pola_data[0]].push_back(std::stoi(pola_data[i]));
                }
                pola_target_vector.push_back({pola_target_map});
            }
            if (pola_quant_map.find(pola_data[0]) == pola_quant_map.end() && !strncmp(pola_data[0].c_str(),"quant",5)){
                for(int i=1;i<pola_data.size();i++){
                    pola_quant_map[pola_data[0]].push_back(std::stoi(pola_data[i]));
                }
                pola_quant_vector.push_back({pola_quant_map});
            }
		}
    file.close();
    //-----------------------------------------------
    return std::tuple<json,json,json,json>(layers,json(pola_target_vector),json(pola_quant_vector),output_layers);
}
void pstruct::scan_table(const json layers,const std::string output_layer){//掃表格，回推去抓concat位置
    size_t count=0;
    std::string cmp_string = "";
    for(auto node : layers){
        if(node["name"]==output_layer){
            //std::cout<<node["name"]<<std::endl;
            if(node["inbound_nodes"][0][0][0].empty()){//代表他是最前面的節點只會走上一次
                break;
            }
            else{
                for(auto node_count : node["inbound_nodes"][0]){
                    auto iter = scan_result_map.find(node_count[0]);
                    if(node["class_name"]=="Concatenate"){//抓出concat所有節點做concat
                        if(node["name"]==cmp_string || cmp_string==""){
                            cmp_string = node["name"];
                            concat_map_vector[node_count[0]] = count;
                            //std::cout<<"up "<<node_count[0]<<std::endl;
                        }
                        else{
                            //std::cout<<"down "<<node_count[0]<<std::endl;
                            cmp_string = node["name"];
                            concat_map_vector[node_count[0]] = ++count;
                        }
                    }
                    if(iter == scan_result_map.end()){
                        scan_result_map[node_count[0]] = 0;
                        scan_table(layers,node_count[0]);
                    }
                    else{
                        scan_result_map[node_count[0]] = 1; //代表這邊的節點已經被走過了
                        scan_result_branch_node_map[node["name"]] = 1;
                        std::cout<<"is been go before , the node is "<<node_count[0]<<" and the branch node is "<<node["name"]<<std::endl;
                        break;
                    }
                }
            }
        }
    }
}
std::vector<json>pstruct::split_layer_info_to_vector(const json &data_info){
    std::vector<json> data_info_vector;
    data_info_vector.reserve(data_info.size());
    for(auto i : data_info){
        data_info_vector.push_back(i);
    }
    return data_info_vector;
}
void pstruct::fc_divd ( int32_t node, int32_t &dimx, int32_t &dimy, int32_t &dimz){
	dimz = ( node <= 24 ) ? node : 24;
	dimx =  (std::ceil(((float)(node))/24));
	std::cout<<"node/24 ="<<dimx<<std::endl;	
	dimy = 1;
    size_t limit = 0;
	while ( dimx > 64 ) { //why 64 因為是2的6次方，這邊只能給到6bit，太大的話要往下切
		for( auto tmp=1 ; tmp<64 ; tmp++ ){
			if ( dimx%tmp == 0 ){
				dimx = dimx/tmp;
				dimy = dimy*tmp;
			}	
		}	
        limit ++;
        if ( limit == 30000 )
            throw std::invalid_argument("Error, Can't Resize Node to 3D");
	}
}
std::vector<std::shared_ptr<pstruct>>pstruct::layer_info_data_vector_trace(const std::vector<json>&layer_info_data_vector){
    std::vector<std::shared_ptr<pstruct>> layer_info_data_pointer_vector;
    std::vector<int32_t>Concat_output_control_location_inside;
    std::vector<int32_t>concat_result_node_list;
    for(auto layer_info : layer_info_data_vector){
        std::cout<<layer_info["class_name"]<<std::endl;
        std::shared_ptr<pstruct> temp = std::make_shared<pstruct>();
        if(layer_info["class_name"]=="InputLayer"){
            temp->class_name          = (!layer_info["class_name"].empty())                     ? layer_info["class_name"]                     : (json)s_none;
            temp->input_feature_size  = (!layer_info["config"]["batch_input_shape"][1].empty()) ? layer_info["config"]["batch_input_shape"][1]   : (json)n_none;
            temp->input_channel       = (!layer_info["config"]["batch_input_shape"][3].empty()) ? layer_info["config"]["batch_input_shape"][3]   : (json)n_none;
        }
        if(layer_info["class_name"]=="Conv2D"){
            temp->class_name          = (!layer_info["class_name"].empty())                     ? layer_info["class_name"]                     : (json)s_none; //s
            temp->input_feature_size  = (!layer_info["config"]["batch_input_shape"][1].empty()) ? layer_info["config"]["batch_input_shape"][1] : (json)n_none;
            temp->input_channel       = (!layer_info["config"]["batch_input_shape"][3].empty()) ? layer_info["config"]["batch_input_shape"][3] : (json)n_none;
            temp->output_channel      = (!layer_info["config"]["filters"].empty())              ? layer_info["config"]["filters"]              : (json)n_none; 
            temp->kernel_size         = (!layer_info["config"]["kernel_size"][0].empty())       ? layer_info["config"]["kernel_size"][0]       : (json)n_none; 
            temp->kernel_stride       = (!layer_info["config"]["strides"][0].empty())           ? layer_info["config"]["strides"][0]           : (json)n_none; 
            temp->padding             = (!layer_info["config"]["padding"].empty())              ? layer_info["config"]["padding"]              : (json)s_none; //s
            temp->activate            = (!layer_info["config"]["activation"].empty())           ? layer_info["config"]["activation"]           : (json)s_none; //s
            temp->Have_bias           = (!layer_info["config"]["use_bias"].empty())             ? layer_info["config"]["use_bias"]             : (json)false ; //bool
            temp->weight_name         = layer_info["config"]["name"];
            temp->units               = n_none; 
            temp->pool_size           = n_none;
            temp->pool_padding        = s_none; //s
            temp->pool_stride         = n_none;
            temp->node_name           = layer_info["config"]["name"];
            temp->Have_ReLU           = temp->activate=="relu" ? true : false;
        }
        if(layer_info["class_name"]=="MaxPooling2D"){
            temp->class_name          = (!layer_info["class_name"].empty())                     ? layer_info["class_name"]                     : (json)s_none; //s
            temp->pool_size           = (!layer_info["config"]["pool_size"][0].empty())         ? layer_info["config"]["pool_size"][0]         : (json)n_none;
            temp->pool_padding        = (!layer_info["config"]["padding"].empty())              ? layer_info["config"]["padding"]              : (json)s_none; //s
            temp->pool_stride         = (!layer_info["config"]["strides"][0].empty())           ? layer_info["config"]["strides"][0]           : (json)n_none;
            temp->input_feature_size  = n_none;
            temp->input_channel       = n_none;
            temp->output_channel      = n_none; 
            temp->kernel_size         = n_none; 
            temp->kernel_stride       = n_none; 
            temp->padding             = s_none; //s
            temp->activate            = s_none; //s
            temp->Have_bias           = false ; //bool
            temp->units               = n_none; 
            temp->Have_Maxpooling     = true;
            temp->node_name           = layer_info["config"]["name"];
        }
        if(layer_info["class_name"]=="Dropout"){
            temp->class_name          = (!layer_info["class_name"].empty())                     ? layer_info["class_name"]                     : (json)s_none; //s
            temp->pool_size           = n_none;
            temp->pool_padding        = s_none; //s
            temp->pool_stride         = n_none;
            temp->input_feature_size  = n_none;
            temp->input_channel       = n_none;
            temp->output_channel      = n_none; 
            temp->kernel_size         = n_none; 
            temp->kernel_stride       = n_none; 
            temp->padding             = s_none; //s
            temp->activate            = s_none; //s
            temp->Have_bias           = false ; //bool
            temp->units               = n_none; 
            temp->node_name           = layer_info["config"]["name"];
        }
        if(layer_info["class_name"]=="Flatten"){
            temp->class_name          = (!layer_info["class_name"].empty())                     ? layer_info["class_name"]                     : (json)s_none; //s
            temp->pool_size           = n_none;
            temp->pool_padding        = s_none; //s
            temp->pool_stride         = n_none;
            temp->input_feature_size  = n_none;
            temp->input_channel       = n_none;
            temp->output_channel      = n_none; 
            temp->kernel_size         = n_none; 
            temp->kernel_stride       = n_none; 
            temp->padding             = s_none; //s
            temp->activate            = s_none; //s
            temp->Have_bias           = false ; //bool
            temp->units               = n_none; 
            temp->Have_Flatten        = true;
            temp->node_name           = layer_info["config"]["name"];
        }
        if(layer_info["class_name"]=="Dense"){
            temp->weight_name         = layer_info["config"]["name"];
            temp->class_name          = (!layer_info["class_name"].empty())                     ? layer_info["class_name"]                     : (json)s_none; //s
            temp->activate            = (!layer_info["config"]["activation"].empty())           ? layer_info["config"]["activation"]           : (json)s_none; //s
            temp->Have_bias           = (!layer_info["config"]["use_bias"].empty())             ? layer_info["config"]["use_bias"]             : (json)false ; //bool
            temp->units               = (!layer_info["config"]["units"].empty())                ? layer_info["config"]["units"]                : (json)n_none; 
            temp->kernel_size         = 1;
            temp->input_feature_size  = n_none;
            temp->input_channel       = n_none;
            temp->output_channel      = n_none;  
            temp->kernel_stride       = n_none; 
            temp->padding             = s_none; //s
            temp->pool_size           = n_none;
            temp->pool_padding        = s_none; //s
            temp->pool_stride         = n_none;
            temp->Have_Dense          = true;
            temp->Have_ReLU           = temp->activate=="relu" ? true : false;
            temp->node_name           = layer_info["config"]["name"];
        }
        if(layer_info["class_name"]=="BatchNormalization"){
            temp->class_name              = (!layer_info["class_name"].empty()) ? layer_info["class_name"]  :   (json)s_none;
            temp->Have_BatchNormalization = true;
            temp->node_name               = layer_info["config"]["name"];
        }
        if(layer_info["class_name"]=="LeakyReLU"){
            temp->class_name              = (!layer_info["class_name"].empty())           ? layer_info["class_name"]  :   (json)s_none;
            temp->Leaky_ReLU_alpha        = (!layer_info["config"]["alpha"].empty())      ? layer_info["config"]["alpha"] : (json)n_none;
            temp->Have_ReLU               = true;
            temp->Is_LeakyReLU            = true;
            temp->node_name               = layer_info["config"]["name"];
        }
        if(layer_info["class_name"]=="UpSampling2D"){
            temp->Have_Upsample           = true;
            temp->class_name              = (!layer_info["class_name"].empty()) ? layer_info["class_name"] : (json)s_none;
            temp->node_name               = layer_info["config"]["name"];
            temp->Upsample_size           = layer_info["config"]["size"][0];
        }
        if(layer_info["class_name"]=="Concatenate"){
            temp->Have_Concat             = true;
            temp->class_name              = layer_info["class_name"];
            temp->node_name               = layer_info["config"]["name"];
        }
        //給前面抓到自己要做分支的節點
        temp->Concat_output_control = (scan_result_map.size()!=0) ? scan_result_map[layer_info["name"]] : 0;
        if(temp->Concat_output_control){
            Concat_output_control_location_inside.push_back((layer_info_data_pointer_vector.size()));//如果有前面有節點要做分支，那就會知道這個位置我要儲存，之後抓weight的時候我們可以直接回推，反之沒有的話這邊本來就應該是-1
        }
        for(auto i : concat_map_vector){
            if(layer_info["name"]==i.first){
                temp->concat_node = 1;
                concat_result_node_list.push_back((layer_info_data_pointer_vector.size()));//如果有前面有節點要做分支，那就會知道這個位置我要儲存，之後抓weight的時候我們可以直接回推，反之沒有的話這邊本來就應該是-1
                std::cout<<"cocat_map_vector "<<temp->node_name<<" "<<temp->concat_node<<" "<<layer_info_data_pointer_vector.size()<<std::endl;
            }
        }
        //給後面去需要前面資訊的節點
        for(auto i : scan_result_branch_node_map){
            if(layer_info["name"]==i.first){
                temp->branch_node = 1;
                if(!temp->Have_Concat){
                    temp->branch_node_location.push_back(Concat_output_control_location_inside[Concat_output_control_location_inside.size()-1]);//確實存在這個地方有分支節點那我就會把我之前丟進caoncat_output_control_location的位置一個個抓出來丟給branch_node_locaiton
                    Concat_output_control_location_inside.pop_back();
                }
                else
                    temp->branch_node_location = concat_result_node_list;
                for(auto i : temp->branch_node_location){
                    std::cout<<temp->node_name<<" "<<i<<std::endl;
                }
            }
        }
        layer_info_data_pointer_vector.push_back(temp);
    }
    return layer_info_data_pointer_vector;
}
void pstruct::weight_offset(std::vector<std::shared_ptr<pstruct>> &layer_info_data_pointer_vector,std::vector<std::vector<int>> &target_layer_info_vector){
    std::vector<int>check_list;
    auto max_weight_data = 0;
    for(int  target=0;target<target_layer_info_vector.size();target++){
        for(auto target_number=0;target_number<target_layer_info_vector[target].size();target_number++){
            auto now_location = target_layer_info_vector[target][target_number];
            if(target==0&&target_number==0){
                if( layer_info_data_pointer_vector[now_location]->class_name=="InputLayer"){
                    layer_info_data_pointer_vector[now_location]->output_channel = layer_info_data_pointer_vector[now_location]->input_channel;
                    layer_info_data_pointer_vector[now_location]->output_feature_size = layer_info_data_pointer_vector[now_location]->input_feature_size;
                    continue;
                }
                //const int32_t Kernel_Size = (3>layer_info_data_pointer_vector[now_location]->kernel_size) ?  3 : layer_info_data_pointer_vector[now_location]->kernel_size;
                const int32_t Kernel_Size   = layer_info_data_pointer_vector[now_location]->kernel_size;
                const int32_t input_channel = (layer_info_data_pointer_vector[now_location]->input_channel==-1) ? throw std::invalid_argument("the input layer channel size is negative") : layer_info_data_pointer_vector[now_location]->input_channel;
                int32_t output_channel = (layer_info_data_pointer_vector[now_location]->output_channel==-1) ? throw std::invalid_argument("the input layer output_channel is negative") : layer_info_data_pointer_vector[now_location]->output_channel;
                int32_t ifs = (layer_info_data_pointer_vector[now_location]->input_feature_size==-1) ? throw std::invalid_argument("the input layer ifs is negative") : layer_info_data_pointer_vector[now_location]->input_feature_size;
                int32_t ofs = 0;
                if(layer_info_data_pointer_vector[now_location]->class_name=="Conv2D"){
                    if(layer_info_data_pointer_vector[now_location]->padding=="valid"){
                        ofs = (Kernel_Size==3) ? (ifs-layer_info_data_pointer_vector[now_location]->kernel_size+1)/layer_info_data_pointer_vector[now_location]->kernel_stride //如果是3的話我的ofs一定會變小 tile_x and tile_y 都會減2 ，但是如果是1*1的話那就都不會有事情
                                               : (ifs)/layer_info_data_pointer_vector[now_location]->kernel_stride;
                        layer_info_data_pointer_vector[now_location]->input_padding_size = 0;
                    }
                    else if(layer_info_data_pointer_vector[now_location]->padding=="same"){
                        ofs = ifs/layer_info_data_pointer_vector[now_location]->kernel_stride;
                        size_t tmp_of_size = (Kernel_Size==3) ? (size_t)std::ceil((ifs-layer_info_data_pointer_vector[now_location]->kernel_size+1)/(double)layer_info_data_pointer_vector[now_location]->kernel_stride)
                                                              : (size_t)std::ceil((ifs)/(double)layer_info_data_pointer_vector[now_location]->kernel_stride);
                        layer_info_data_pointer_vector[now_location]->input_padding_size = (ofs-tmp_of_size)/2;
                    }
                    else if(layer_info_data_pointer_vector[now_location]->padding=="none"){
                        ofs = ifs;
                    }
                    else throw std::invalid_argument("error we dont have this padding option");
                }else{
                    throw std::invalid_argument("sorry your first layer not conv we got some error please check your json file or check your first node is not conv");
                }
                size_t weight_of_kernel,weight_of_AB;
                if(Kernel_Size==3){
                    weight_of_kernel = round_ch_8(input_channel)*round_ch_8(output_channel)*std::pow(Kernel_Size,2)*DATA_DEPTH*DATA_DEPTH*DATA_WIDTH;
                    weight_of_AB     = round_ch_8(output_channel)*DATA_DEPTH*2*DATA_WIDTH;
                }else if(Kernel_Size==1){
                    weight_of_kernel = round_ch_24(input_channel)*round_ch_24(output_channel)*std::pow(Kernel_Size,2)*DATA_Depth_24*DATA_Depth_24*DATA_WIDTH;
                    weight_of_AB     = round_ch_24(output_channel)*DATA_Depth_24*2*DATA_WIDTH;
                }else{
                    std::cout<<Kernel_Size<<std::endl;
                    std::cerr<<"sorry we dont have this option，you got something wrong at function weight_offset"<<std::endl;
                }
                layer_info_data_pointer_vector[now_location]->weight_address = max_weight_data;
                max_weight_data = weight_of_kernel + weight_of_AB;
                layer_info_data_pointer_vector[now_location]->input_feature_size  = ifs;
                layer_info_data_pointer_vector[now_location]->output_feature_size = ofs;
            }else{
                size_t pre_location = 0;
                int32_t input_channel = 0;
                std::vector<size_t>pre_location_vec;
                if(!layer_info_data_pointer_vector[now_location]->branch_node){
                    pre_location = target>0 ? (target_number==0 ? target_layer_info_vector[target-1][target_layer_info_vector[target-1].size()-1] : target_layer_info_vector[target][target_number-1]) : target_layer_info_vector[target][target_number-1];//pre target merge last one node
                    pre_location_vec.push_back(pre_location);    
                }else{
                    if(layer_info_data_pointer_vector[now_location]->Have_Concat){
                        for(auto index : layer_info_data_pointer_vector[now_location]->branch_node_location){
                            pre_location_vec.push_back(index);
                            std::cout<<"i am concat i am good value is "<<index<<std::endl;
                        }
                    }
                    else
                        pre_location_vec.push_back(layer_info_data_pointer_vector[now_location]->branch_node_location[0]);
                    for(auto index : pre_location_vec){
                        std::cout<<"branch_node got it "<<index<<std::endl;
                    }
                }
                //const int32_t Kernel_Size   = (3>layer_info_data_pointer_vector[now_location]->kernel_size) ?  3 : layer_info_data_pointer_vector[now_location]->kernel_size;
                const int32_t Kernel_Size   = layer_info_data_pointer_vector[now_location]->kernel_size; //以前是不管怎樣我都用3*3來算，小於3我也是3但是現在我要改成有3*3和1*1
                if(!layer_info_data_pointer_vector[now_location]->Have_Concat)
                    input_channel = (layer_info_data_pointer_vector[now_location]->input_channel==-1) ? layer_info_data_pointer_vector[pre_location_vec[0]]->output_channel : layer_info_data_pointer_vector[now_location]->input_channel;
                else{
                    for(auto index : pre_location_vec){
                        input_channel += layer_info_data_pointer_vector[index]->output_channel;
                        std::cout<<"CHECK!!!!! PRE_NODE~~~~~"<<layer_info_data_pointer_vector[index]->output_channel<<std::endl;
                    }
                }
                for(auto index : pre_location_vec){
                    std::cout<<" there is input_channel "<<input_channel<<" now_location "<<now_location<<" pre_location "<<index<<std::endl;
                }
                int32_t output_channel = 0;
                int32_t ifs = layer_info_data_pointer_vector[now_location]->input_feature_size==-1 ? layer_info_data_pointer_vector[pre_location_vec[0]]->output_feature_size : layer_info_data_pointer_vector[now_location]->input_feature_size;
                int32_t ofs = 0;
                if(layer_info_data_pointer_vector[now_location]->class_name=="Dense"){
                    output_channel = layer_info_data_pointer_vector[now_location]->units;
                    layer_info_data_pointer_vector[now_location]->Have_Dense = true;
                }
                else if(layer_info_data_pointer_vector[now_location]->class_name=="Flatten"){
                    layer_info_data_pointer_vector[now_location]->Have_Flatten = true;
                    output_channel = layer_info_data_pointer_vector[now_location]->output_channel==-1 ? input_channel : layer_info_data_pointer_vector[now_location]->output_channel;
                }
                else{
                    output_channel = layer_info_data_pointer_vector[now_location]->output_channel==-1 ? input_channel : layer_info_data_pointer_vector[now_location]->output_channel;
                }
                layer_info_data_pointer_vector[now_location]->output_channel = output_channel;
                //std::cout<<layer_info_data_pointer_vector[now_location]->output_channel<<std::endl;
                if(layer_info_data_pointer_vector[now_location]->class_name=="Conv2D"){
                    if(layer_info_data_pointer_vector[now_location]->padding=="valid"){
                        ofs = (Kernel_Size==3) ? (ifs-layer_info_data_pointer_vector[now_location]->kernel_size+1)/layer_info_data_pointer_vector[now_location]->kernel_stride //如果是3的話我的ofs一定會變小 tile_x and tile_y 都會減2 ，但是如果是1*1的話那就都不會有事情
                                               : (ifs)/layer_info_data_pointer_vector[now_location]->kernel_stride;
                        layer_info_data_pointer_vector[now_location]->input_padding_size = 0;
                    }
                    else if(layer_info_data_pointer_vector[now_location]->padding=="same"){
                        ofs = ifs/layer_info_data_pointer_vector[now_location]->kernel_stride;
                        size_t tmp_of_size = (Kernel_Size==3) ? (size_t)std::ceil((ifs-layer_info_data_pointer_vector[now_location]->kernel_size+1)/(double)layer_info_data_pointer_vector[now_location]->kernel_stride)
                                                              : (size_t)std::ceil((ifs)/(double)layer_info_data_pointer_vector[now_location]->kernel_stride);
                        layer_info_data_pointer_vector[now_location]->input_padding_size = (ofs-tmp_of_size)/2;
                    }
                    else if(layer_info_data_pointer_vector[now_location]->padding=="none"){
                        ofs = ifs;
                    }
                    else throw std::invalid_argument("error we dont have this padding option");
                }
                else if(layer_info_data_pointer_vector[now_location]->class_name=="MaxPooling2D"){
                    layer_info_data_pointer_vector[now_location]->Have_Maxpooling = true;
                    if(layer_info_data_pointer_vector[now_location]->pool_padding=="valid"){
                        ofs = std::ceil((ifs-layer_info_data_pointer_vector[now_location]->pool_size+1)/(double)layer_info_data_pointer_vector[now_location]->pool_stride);
                    }
                    else if(layer_info_data_pointer_vector[now_location]->pool_padding=="same"){
                        ofs = ifs/layer_info_data_pointer_vector[now_location]->pool_stride;
                    }
                    else if(layer_info_data_pointer_vector[now_location]->pool_padding=="none"){
                        ofs = ifs;
                    }
                    else std::cerr<<"error we dont have this padding option"<<std::endl;
                }
                else if(layer_info_data_pointer_vector[now_location]->class_name=="Dense"){
                    ofs = 1; //FC出來的output_feature_size都是1
                }
                else ofs = ifs;
                size_t weight_of_kernel,weight_of_AB;
                if(layer_info_data_pointer_vector[now_location]->class_name=="Dense"){
                    if(layer_info_data_pointer_vector[pre_location_vec[0]]->class_name=="Dense"){
                        layer_info_data_pointer_vector[now_location]->IF_PRE_NODE_IS_DENSE = true;
                        //layer_info_data_pointer_vector[now_location]->Dense_input_node = layer_info_data_pointer_vector[pre_location]->Dense_output_channel * 
                        //                                                                 layer_info_data_pointer_vector[pre_location]->Dense_size_output_x  * 
                        //                                                                 layer_info_data_pointer_vector[pre_location]->Dense_size_output_y  
                        //                                                                 ;
                        layer_info_data_pointer_vector[now_location]->Dense_input_node = layer_info_data_pointer_vector[pre_location]->Dense_output_node;
                        fc_divd(layer_info_data_pointer_vector[now_location]->Dense_input_node,
                                layer_info_data_pointer_vector[now_location]->Dense_size_input_x,
                                layer_info_data_pointer_vector[now_location]->Dense_size_input_y,
                                layer_info_data_pointer_vector[now_location]->Dense_input_channel);
                    }
                    else{
                        layer_info_data_pointer_vector[now_location]->IF_PRE_NODE_IS_DENSE = false;
                        layer_info_data_pointer_vector[now_location]->Dense_size_input_x = (MAX_TILE_SIZE > ifs) ? ifs : MAX_TILE_SIZE;
                        layer_info_data_pointer_vector[now_location]->Dense_size_input_y = (MAX_TILE_SIZE > ifs) ? ifs : MAX_TILE_SIZE;
                        layer_info_data_pointer_vector[now_location]->Dense_input_channel = input_channel;
                        layer_info_data_pointer_vector[now_location]->Dense_input_node = layer_info_data_pointer_vector[now_location]->Dense_input_channel * layer_info_data_pointer_vector[now_location]->Dense_size_input_x * layer_info_data_pointer_vector[now_location]->Dense_size_input_y;
                    }
                    fc_divd(output_channel,
                            layer_info_data_pointer_vector[now_location]->Dense_size_output_x,
                            layer_info_data_pointer_vector[now_location]->Dense_size_output_y,
                            layer_info_data_pointer_vector[now_location]->Dense_output_channel)
                            ;
                    layer_info_data_pointer_vector[now_location]->Dense_output_node = layer_info_data_pointer_vector[now_location]->output_channel ;//* layer_info_data_pointer_vector[now_location]->Dense_size_output_x * layer_info_data_pointer_vector[now_location]->Dense_size_output_y;
                    weight_of_kernel =  round_ch_24(layer_info_data_pointer_vector[now_location]->Dense_output_channel) * 
                                        round_ch_24(layer_info_data_pointer_vector[now_location]->Dense_input_channel) * 
                                        ONCE_CAL_WEIGHT_NUM_1 * 
                                        layer_info_data_pointer_vector[now_location]->Dense_size_input_x  * 
                                        layer_info_data_pointer_vector[now_location]->Dense_size_input_y  * 
                                        layer_info_data_pointer_vector[now_location]->Dense_size_output_x * 
                                        layer_info_data_pointer_vector[now_location]->Dense_size_output_y *
                                        DATA_WIDTH
                                        ;
                    weight_of_AB     =  round_ch_24(layer_info_data_pointer_vector[now_location]->Dense_output_channel) * DATA_Depth_24 * 2 *
                                        layer_info_data_pointer_vector[now_location]->Dense_size_output_x *
                                        layer_info_data_pointer_vector[now_location]->Dense_size_output_y *
                                        DATA_WIDTH
                                        ;
                }else if(Kernel_Size==3){
                    weight_of_kernel = round_ch_8(input_channel)*round_ch_8(output_channel)*std::pow(Kernel_Size,2)*DATA_DEPTH*DATA_DEPTH*DATA_WIDTH;
                    weight_of_AB     = round_ch_8(output_channel)*DATA_DEPTH*2*DATA_WIDTH;
                }else if(Kernel_Size==1){
                    weight_of_kernel = round_ch_24(input_channel)*round_ch_24(output_channel)*std::pow(Kernel_Size,2)*DATA_Depth_24*DATA_Depth_24*DATA_WIDTH;
                    weight_of_AB     = round_ch_24(output_channel)*DATA_Depth_24*2*DATA_WIDTH;
                }else if(Kernel_Size==-1){
                    //donothing ， because you only got Kernel_size in CONV
                }else{
                    std::cerr<<"sorry we dont have this option，you got something wrong at function weight_offset"<<std::endl;
                }
                if(layer_info_data_pointer_vector[now_location]->class_name=="Conv2D"||layer_info_data_pointer_vector[now_location]->class_name=="Dense"){
                    for(int  target=0;target<target_layer_info_vector.size();target++){
                        for(auto target_number=0;target_number<target_layer_info_vector[target].size();target_number++){
                            max_weight_data = (layer_info_data_pointer_vector[target_layer_info_vector[target][target_number]]->weight_address > max_weight_data) ? layer_info_data_pointer_vector[target_layer_info_vector[target][target_number]]->weight_address : max_weight_data;
                        }
                    }
                    layer_info_data_pointer_vector[now_location]->weight_address =  max_weight_data;
                    //std::cout<<"max_weight_data "<<max_weight_data<<" weight_of_AB "<<weight_of_AB<<" weight_of_kernel "<<weight_of_kernel<<std::endl;
                    max_weight_data+=weight_of_kernel + weight_of_AB;
                }
                layer_info_data_pointer_vector[now_location]->input_channel = input_channel;
                layer_info_data_pointer_vector[now_location]->input_feature_size  = ifs;
                layer_info_data_pointer_vector[now_location]->output_feature_size = ofs;
            }
        }
    }
}
std::vector<std::vector<int>>target_node_trace(json &target_info){
    std::vector<std::vector<int>>target_layer_info_vector;
    for(int i=0;i<target_info.size();i++){
        std::string tmp = "layers_" + std::to_string(i);
        target_layer_info_vector.push_back(target_info[i][tmp]);
    }
    return target_layer_info_vector;
}
std::vector<std::vector<int>>target_quant_trace(json &target_quant){
    std::vector<std::vector<int>>target_layer_quant_vector;
    for(int i=0;i<target_quant.size();i++){
        std::string tmp = "quant_" + std::to_string(i);
        target_layer_quant_vector.push_back(target_quant[i][tmp]);
    }
    return target_layer_quant_vector;
}

std::vector<std::shared_ptr<pstruct>>pstruct::merge_node(std::vector<std::shared_ptr<pstruct>> &layer_info_data_pointer_vector,std::vector<std::vector<int>> &target_layer_info_vector,std::vector<std::vector<int>> &target_layer_quant_vector,std::vector<int>&merge_node_jump_location){
    std::vector<std::shared_ptr<pstruct>>merge_node_vector;
    for(int  target=0;target<target_layer_info_vector.size();target++){
        std::shared_ptr<pstruct>new_pointer = std::make_shared<pstruct>();
        for(auto target_number=0;target_number<target_layer_info_vector[target].size();target_number++){
            new_pointer->node_name            = (new_pointer->weight_name=="none")                ? layer_info_data_pointer_vector[target_layer_info_vector[target][target_number]]->node_name               : new_pointer->node_name;
            new_pointer->weight_name          = (new_pointer->weight_name=="none")                ? layer_info_data_pointer_vector[target_layer_info_vector[target][target_number]]->weight_name             : new_pointer->weight_name;
            new_pointer->class_name           = new_pointer->class_name=="none"                   ? layer_info_data_pointer_vector[target_layer_info_vector[target][target_number]]->class_name              : new_pointer->class_name+" + "+layer_info_data_pointer_vector[target_layer_info_vector[target][target_number]]->class_name;
            new_pointer->input_channel        = new_pointer->input_channel==-1                    ? layer_info_data_pointer_vector[target_layer_info_vector[target][target_number]]->input_channel           : new_pointer->input_channel;
            new_pointer->input_feature_size   = new_pointer->input_feature_size==-1               ? layer_info_data_pointer_vector[target_layer_info_vector[target][target_number]]->input_feature_size      : new_pointer->input_feature_size;
            new_pointer->output_channel       = layer_info_data_pointer_vector[target_layer_info_vector[target][target_number]]->output_channel; //your output_channel will be replace when the next node
            new_pointer->output_feature_size  = layer_info_data_pointer_vector[target_layer_info_vector[target][target_number]]->output_feature_size;//your output_feature size will be replace when the next node
            new_pointer->kernel_size          = (new_pointer->kernel_size==-1)                    ? layer_info_data_pointer_vector[target_layer_info_vector[target][target_number]]->kernel_size             : new_pointer->kernel_size;
            new_pointer->kernel_stride        = (new_pointer->kernel_stride==-1)                  ? layer_info_data_pointer_vector[target_layer_info_vector[target][target_number]]->kernel_stride           : new_pointer->kernel_stride;
            new_pointer->padding              = (new_pointer->padding =="none")                   ? layer_info_data_pointer_vector[target_layer_info_vector[target][target_number]]->padding                 : new_pointer->padding;
            new_pointer->activate             = (new_pointer->activate=="none")                   ? layer_info_data_pointer_vector[target_layer_info_vector[target][target_number]]->activate                : new_pointer->activate;
            new_pointer->pool_size            = (new_pointer->pool_size==-1)                      ? layer_info_data_pointer_vector[target_layer_info_vector[target][target_number]]->pool_size               : new_pointer->pool_size;
            new_pointer->pool_stride          = (new_pointer->pool_stride==-1)                    ? layer_info_data_pointer_vector[target_layer_info_vector[target][target_number]]->pool_stride             : new_pointer->pool_stride;
            new_pointer->pool_padding         = (new_pointer->pool_padding=="none")               ? layer_info_data_pointer_vector[target_layer_info_vector[target][target_number]]->pool_padding            : new_pointer->pool_padding;
            new_pointer->units                = (new_pointer->units==-1)                          ? layer_info_data_pointer_vector[target_layer_info_vector[target][target_number]]->units                   : new_pointer->units;
            new_pointer->Have_bias            = (new_pointer->Have_bias==false)                   ? layer_info_data_pointer_vector[target_layer_info_vector[target][target_number]]->Have_bias               : new_pointer->Have_bias;
            new_pointer->weight_address       = (new_pointer->weight_address==-1)                 ? layer_info_data_pointer_vector[target_layer_info_vector[target][target_number]]->weight_address          : new_pointer->weight_address;
            new_pointer->input_padding_size   = (new_pointer->input_padding_size==0)              ? layer_info_data_pointer_vector[target_layer_info_vector[target][target_number]]->input_padding_size      : new_pointer->input_padding_size;
            new_pointer->Leaky_ReLU_alpha     = (new_pointer->Leaky_ReLU_alpha==-1)               ? layer_info_data_pointer_vector[target_layer_info_vector[target][target_number]]->Leaky_ReLU_alpha        : new_pointer->Leaky_ReLU_alpha;
            new_pointer->Have_BatchNormalization = (new_pointer->Have_BatchNormalization==false)  ? layer_info_data_pointer_vector[target_layer_info_vector[target][target_number]]->Have_BatchNormalization : new_pointer->Have_BatchNormalization;
            new_pointer->Have_Flatten         = (new_pointer->Have_Flatten==false)                ? layer_info_data_pointer_vector[target_layer_info_vector[target][target_number]]->Have_Flatten            : new_pointer->Have_Flatten   ; 
            new_pointer->Have_Dense           = (new_pointer->Have_Dense==false)                  ? layer_info_data_pointer_vector[target_layer_info_vector[target][target_number]]->Have_Dense              : new_pointer->Have_Dense     ; 
            new_pointer->Have_Maxpooling      = (new_pointer->Have_Maxpooling==false)             ? layer_info_data_pointer_vector[target_layer_info_vector[target][target_number]]->Have_Maxpooling         : new_pointer->Have_Maxpooling; 
            new_pointer->Is_LeakyReLU         = (new_pointer->Is_LeakyReLU==false)                ? layer_info_data_pointer_vector[target_layer_info_vector[target][target_number]]->Is_LeakyReLU            : new_pointer->Is_LeakyReLU; 
            new_pointer->Have_Upsample        = (new_pointer->Have_Upsample==false)               ? layer_info_data_pointer_vector[target_layer_info_vector[target][target_number]]->Have_Upsample           : new_pointer->Have_Upsample;
            new_pointer->Have_Concat          = (new_pointer->Have_Concat==false)                 ? layer_info_data_pointer_vector[target_layer_info_vector[target][target_number]]->Have_Concat             : new_pointer->Have_Concat;
            new_pointer->Upsample_size        = (new_pointer->Upsample_size==0)                   ? layer_info_data_pointer_vector[target_layer_info_vector[target][target_number]]->Upsample_size           : new_pointer->Upsample_size;
            new_pointer->Concat_output_control= (new_pointer->Concat_output_control==false)       ? layer_info_data_pointer_vector[target_layer_info_vector[target][target_number]]->Concat_output_control   : new_pointer->Concat_output_control;
            if(layer_info_data_pointer_vector[target_layer_info_vector[target][target_number]]->class_name=="BatchNormalization" && new_pointer->Have_ReLU==false){
                new_pointer->Batch_First = true;
            }
            if(layer_info_data_pointer_vector[target_layer_info_vector[target][target_number]]->class_name=="Dense"){
                new_pointer->Batch_First = true;
            }
            new_pointer->Have_ReLU            = (new_pointer->Have_ReLU==false)                   ? layer_info_data_pointer_vector[target_layer_info_vector[target][target_number]]->Have_ReLU               : new_pointer->Have_ReLU      ; 
            new_pointer->branch_node          = (new_pointer->branch_node==false)                 ? layer_info_data_pointer_vector[target_layer_info_vector[target][target_number]]->branch_node             : new_pointer->branch_node;
            new_pointer->concat_node          = (new_pointer->concat_node==false)                 ? layer_info_data_pointer_vector[target_layer_info_vector[target][target_number]]->concat_node             : new_pointer->concat_node;
            new_pointer->Dense_input_channel  = (new_pointer->Dense_input_channel==false)         ? layer_info_data_pointer_vector[target_layer_info_vector[target][target_number]]->Dense_input_channel     : new_pointer->Dense_input_channel ;
            new_pointer->Dense_input_node     = (new_pointer->Dense_input_node==false)            ? layer_info_data_pointer_vector[target_layer_info_vector[target][target_number]]->Dense_input_node        : new_pointer->Dense_input_node    ;
            new_pointer->Dense_output_channel = (new_pointer->Dense_output_channel==false)        ? layer_info_data_pointer_vector[target_layer_info_vector[target][target_number]]->Dense_output_channel    : new_pointer->Dense_output_channel;
            new_pointer->Dense_output_node    = (new_pointer->Dense_output_node==false)           ? layer_info_data_pointer_vector[target_layer_info_vector[target][target_number]]->Dense_output_node       : new_pointer->Dense_output_node   ;
            new_pointer->Dense_size_input_x   = (new_pointer->Dense_size_input_x==false)          ? layer_info_data_pointer_vector[target_layer_info_vector[target][target_number]]->Dense_size_input_x      : new_pointer->Dense_size_input_x  ;
            new_pointer->Dense_size_input_y   = (new_pointer->Dense_size_input_y==false)          ? layer_info_data_pointer_vector[target_layer_info_vector[target][target_number]]->Dense_size_input_y      : new_pointer->Dense_size_input_y  ;
            new_pointer->Dense_size_output_x  = (new_pointer->Dense_size_output_x==false)         ? layer_info_data_pointer_vector[target_layer_info_vector[target][target_number]]->Dense_size_output_x     : new_pointer->Dense_size_output_x ;
            new_pointer->Dense_size_output_y  = (new_pointer->Dense_size_output_y==false)         ? layer_info_data_pointer_vector[target_layer_info_vector[target][target_number]]->Dense_size_output_y     : new_pointer->Dense_size_output_y ;
            new_pointer->IF_PRE_NODE_IS_DENSE = (new_pointer->IF_PRE_NODE_IS_DENSE==false)        ? layer_info_data_pointer_vector[target_layer_info_vector[target][target_number]]->IF_PRE_NODE_IS_DENSE    : new_pointer->IF_PRE_NODE_IS_DENSE;
            new_pointer->node_name_vector.push_back(new_pointer->node_name);
        }
        //----------------------下面兩個差非常多喔~~~~----------------------------
        if(new_pointer->concat_node){//這邊不能用concat_output_control因為這個東西是和硬體說我要做合併，因此我需要知道最後的哪個點合併，中間說不定有更多東西還要做
            merge_node_jump_location.push_back(merge_node_vector.size());
            std::cout<<"merge_node_jump_location.size"<<merge_node_vector.size()<<std::endl;
        }
        if(new_pointer->Concat_output_control){//這邊是你要抓出來你的branch的節點，因為你要透過這個去知道說你的input在哪邊，這邊是一分之就要記錄
            branch_node_jump_location.push_back(merge_node_vector.size());
            std::cout<<"branch_node_jump_location.size"<<merge_node_vector.size()<<std::endl;
        }
        //----------------------上面兩個差非常多喔~~~~----------------------------
        new_pointer->output_feature_size = (new_pointer->Have_Upsample) ? new_pointer->output_feature_size*2 : new_pointer->output_feature_size;
        merge_node_vector.push_back(new_pointer);
    }
    for(int quant_list_index = 0;quant_list_index<target_layer_quant_vector.size();quant_list_index++){
        merge_node_vector[quant_list_index]->quant_batch           = target_layer_quant_vector[quant_list_index][0];
        merge_node_vector[quant_list_index]->quant_batch_bias      = target_layer_quant_vector[quant_list_index][1];
        merge_node_vector[quant_list_index]->quant_finish          = target_layer_quant_vector[quant_list_index][2];
        merge_node_vector[quant_list_index]->quant_obuf            = target_layer_quant_vector[quant_list_index][3];
        merge_node_vector[quant_list_index]->quant_word_size       = target_layer_quant_vector[quant_list_index][4];
        merge_node_vector[quant_list_index]->pooling_quant_finish  = target_layer_quant_vector[quant_list_index][5];
    }
    return merge_node_vector;
}
void pstruct::merge_node_fix(std::vector<std::shared_ptr<pstruct>> &merge_node_vector,std::vector<int>&merge_node_jump_location,std::vector<int>&branch_node_jump_location){
    std::vector<int32_t>offset_vector;
    std::vector<int>branch_node_jump_location_inside = branch_node_jump_location;
    for(int i=0;i<merge_node_vector.size();i++){
        if(i==0){//first node
            const size_t Depth_select = merge_node_vector[i]->class_name=="Dense" ?  DATA_Depth_24 : merge_node_vector[i]->kernel_size==1 ? DATA_Depth_24 : DATA_DEPTH;
            merge_node_vector[i]->node_name              = merge_node_vector[i]->weight_name;//+"_"+std::to_string(i+1);
            merge_node_vector[i]->weight_name            = merge_node_vector[i]->weight_name;//+"_"+std::to_string(i+1);
            merge_node_vector[i]->output_padding_size    = merge_node_vector[i]->padding=="same" ? merge_node_vector[i]->input_padding_size : 0;
            merge_node_vector[i]->input_tile_number      = calc_input_tile_number(merge_node_vector[i]);
            merge_node_vector[i]->input_tile_size        = calc_layer_tile_size(merge_node_vector[i]);
            merge_node_vector[i]->next_tile_size         = calc_next_layer_tile_size(merge_node_vector[i]);
            merge_node_vector[i]->branch_input_tile_size = calc_branch_next_layer_tile_size(merge_node_vector[i]);
            merge_node_vector[i]->output_tile_size       = (merge_node_vector[i]->Have_Upsample) ? merge_node_vector[i]->Upsample_size * calc_output_layer_tile_size(merge_node_vector[i]) : calc_output_layer_tile_size(merge_node_vector[i]);
            merge_node_vector[i]->output_tile_number     = calc_output_tile_number(merge_node_vector[i]);
            const size_t layer_input_ch                  = std::ceil(merge_node_vector[i]->input_channel/(double)Depth_select)*Depth_select;
            const size_t output_size                     = std::pow(merge_node_vector[i]->input_tile_number,2)*std::pow(merge_node_vector[i]->input_tile_size,2)*layer_input_ch*DATA_WIDTH;
            merge_node_vector[i]->input_address.push_back(0);
            if(merge_node_vector[i]->Have_Maxpooling){//TODO
                merge_node_vector[i]->pool_address     = merge_node_vector[i]->input_address[0] + ((merge_node_vector[i]->Have_Upsample) ? std::pow(merge_node_vector[i]->Upsample_size,2)*output_size : output_size);
            }
            else{
                merge_node_vector[i]->output_address   = merge_node_vector[i]->input_address[0] + ((merge_node_vector[i]->Have_Upsample) ? std::pow(merge_node_vector[i]->Upsample_size,2)*output_size : output_size);
            }
            merge_node_vector[i]->Previous_node_OCH.push_back(merge_node_vector[i]->input_channel); 
            offset_vector.push_back(merge_node_vector[i]->output_address);
        }
        else if(merge_node_vector[i]->Have_Concat){//TODO
            std::cout<<"Concatenate now!!!!!!!!!!!!!!!!!!!"<<std::endl;
            const size_t Depth_select = merge_node_vector[i]->class_name=="Dense" ?  DATA_Depth_24 : merge_node_vector[i]->kernel_size==1 ? DATA_Depth_24 : DATA_DEPTH;
            size_t address_select,branch_output_address_select;
            size_t merge_node_jump_location_select=0,merge_output_channel=0;
            merge_node_vector[i]->node_name              = merge_node_vector[i]->weight_name;//+"_"+std::to_string(i+1);
            merge_node_vector[i]->weight_name            = merge_node_vector[i]->weight_name;//+"_"+std::to_string(i+1);
            merge_node_vector[i]->output_padding_size    = merge_node_vector[i]->padding=="same" ? merge_node_vector[i-1]->input_padding_size : 0;
            merge_node_vector[i]->input_tile_number      = calc_input_tile_number(merge_node_vector[i]);
            merge_node_vector[i]->input_tile_size        = calc_layer_tile_size(merge_node_vector[i]);
            merge_node_vector[i]->next_tile_size         = calc_next_layer_tile_size(merge_node_vector[i]);
            merge_node_vector[i]->branch_input_tile_size = calc_branch_next_layer_tile_size(merge_node_vector[i]);
            merge_node_vector[i]->output_tile_size       = (merge_node_vector[i]->Have_Upsample) ? merge_node_vector[i]->Upsample_size * calc_output_layer_tile_size(merge_node_vector[i]) : calc_output_layer_tile_size(merge_node_vector[i]);
            merge_node_vector[i]->output_tile_number     = calc_output_tile_number(merge_node_vector[i]);
            for(int num = merge_node_jump_location.size()-1;num>=0;num--){
                merge_node_jump_location_select =  merge_node_jump_location[num]; //最後輸出merge_node_jump_location_select = 4;為啥要這個就要看訓練的人怎麼擺了
                merge_output_channel            += merge_node_vector[merge_node_jump_location_select]->output_channel;
                address_select                  =  merge_node_vector[merge_node_jump_location_select]->output_address;//上一層的輸出，這一層的輸入
                merge_node_vector[i]->input_address.push_back(address_select/*offset_vector[std::stoi(tmp)-2]*/);//this 2 is preset the count number
                merge_node_vector[i]->Previous_node_OCH.push_back(merge_node_vector[merge_node_jump_location_select]->output_channel);
            }
            merge_node_vector[i]->input_channel        = merge_output_channel; //這邊Input要改成兩邊相加後的結果
            const size_t layer_input_ch                = std::ceil(merge_node_vector[i]->input_channel/(double)Depth_select)*Depth_select;
            const size_t output_size                   = std::pow(merge_node_vector[merge_node_jump_location_select]->input_tile_number,2)*std::pow(merge_node_vector[merge_node_jump_location_select]->input_tile_size,2)*layer_input_ch*DATA_WIDTH;
            branch_output_address_select  = merge_node_vector[i-1]->Have_Maxpooling ? merge_node_vector[i-1]->pool_address : merge_node_vector[i-1]->output_address;//這邊一定要抓最後的node因為要推weight
            if(merge_node_vector[i]->Concat_output_control && merge_node_vector[i]->Have_Maxpooling){
                const size_t layer_output_ch             = std::ceil(merge_node_vector[i]->output_channel/(double)Depth_select)*Depth_select;
                auto pool_address_calc = std::pow(merge_node_vector[i]->input_tile_number,2)*std::pow(merge_node_vector[i]->input_tile_size,2)*layer_output_ch*DATA_WIDTH;
                merge_node_vector[i]->output_address     = branch_output_address_select + ((merge_node_vector[i]->Have_Upsample) ? std::pow(merge_node_vector[i]->Upsample_size,2)*output_size : output_size);
                merge_node_vector[i]->pool_address       = branch_output_address_select + ((merge_node_vector[i]->Have_Upsample) ? std::pow(merge_node_vector[i]->Upsample_size,2)*pool_address_calc : pool_address_calc);
            }
            else if(merge_node_vector[i]->Have_Maxpooling){
                merge_node_vector[i]->pool_address       = branch_output_address_select + ((merge_node_vector[i]->Have_Upsample) ? std::pow(merge_node_vector[i]->Upsample_size,2)*output_size : output_size); //這邊我要說的是，我前面推完output_address但是我要做concat，那我就會把資料丟在我之前做output_address後面再去加上一次因為我要丟兩個地方
            }
            else if(!merge_node_vector[i]->Have_Maxpooling){
                merge_node_vector[i]->output_address     = branch_output_address_select + ((merge_node_vector[i]->Have_Upsample) ? std::pow(merge_node_vector[i]->Upsample_size,2)*output_size : output_size); 
            }else{
                std::cerr<<"sorry we dont have this node"<<std::endl;
            }
        }
        else if(merge_node_vector[i]->Have_Dense){
            std::cout<<"Dense now!!!!!!!!!!!!!!!!!!!!!!!!!"<<std::endl;
            const size_t Depth_select = DATA_Depth_24;//在DENSE 我們通常都是以24為一組來看
            merge_node_vector[i]->node_name              = merge_node_vector[i]->weight_name;
            merge_node_vector[i]->weight_name            = merge_node_vector[i]->weight_name;
            merge_node_vector[i]->output_padding_size    = merge_node_vector[i]->padding=="same" ? merge_node_vector[i]->input_padding_size : 0;
            merge_node_vector[i]->input_tile_number      = calc_dense_input_tile_number(merge_node_vector[i]);
            merge_node_vector[i]->input_tile_size        = calc_dense_layer_tile_size(merge_node_vector[i]);
            merge_node_vector[i]->next_tile_size         = calc_dense_next_layer_tile_size(merge_node_vector[i]);
            merge_node_vector[i]->branch_input_tile_size = calc_dense_branch_next_layer_tile_size(merge_node_vector[i]);
            merge_node_vector[i]->output_tile_size       = (merge_node_vector[i]->Have_Upsample) ? merge_node_vector[i]->Upsample_size * calc_dense_output_layer_tile_size(merge_node_vector[i]) : calc_dense_output_layer_tile_size(merge_node_vector[i]);
            merge_node_vector[i]->output_tile_number     = calc_dense_output_tile_number(merge_node_vector[i]);
            const size_t output_size                     = /*這邊的input_tile_size包含了全部的資料*/merge_node_vector[i]->input_tile_number * std::pow(merge_node_vector[i]->input_tile_size,2) * DATA_WIDTH;
            size_t address_select  = merge_node_vector[i-1]->Have_Maxpooling ? merge_node_vector[i-1]->pool_address : merge_node_vector[i-1]->output_address;
            merge_node_vector[i]->input_address.push_back(address_select);
            merge_node_vector[i]->output_address         = merge_node_vector[i]->input_address[0] + output_size;
        }
        else{
            const size_t Depth_select = merge_node_vector[i]->class_name=="Dense" ?  DATA_Depth_24 : merge_node_vector[i]->kernel_size==1 ? DATA_Depth_24 : DATA_DEPTH;
            merge_node_vector[i]->node_name              = merge_node_vector[i]->weight_name;//+"_"+std::to_string(i+1);
            merge_node_vector[i]->weight_name            = merge_node_vector[i]->weight_name;//+"_"+std::to_string(i+1);
            merge_node_vector[i]->output_padding_size    = merge_node_vector[i]->padding=="same" ? merge_node_vector[i]->input_padding_size : 0;
            merge_node_vector[i]->input_tile_number      = calc_input_tile_number(merge_node_vector[i]);
            merge_node_vector[i]->input_tile_size        = calc_layer_tile_size(merge_node_vector[i]);
            merge_node_vector[i]->next_tile_size         = calc_next_layer_tile_size(merge_node_vector[i]);
            merge_node_vector[i]->branch_input_tile_size = calc_branch_next_layer_tile_size(merge_node_vector[i]);
            merge_node_vector[i]->output_tile_size       = (merge_node_vector[i]->Have_Upsample) ? merge_node_vector[i]->Upsample_size * calc_output_layer_tile_size(merge_node_vector[i]) : calc_output_layer_tile_size(merge_node_vector[i]);
            merge_node_vector[i]->output_tile_number     = calc_output_tile_number(merge_node_vector[i]);
            const size_t layer_input_ch                  = std::ceil(merge_node_vector[i]->input_channel/(double)Depth_select)*Depth_select;
            const size_t output_size                     = std::pow(merge_node_vector[i]->input_tile_number,2)*std::pow(merge_node_vector[i]->input_tile_size,2)*layer_input_ch*DATA_WIDTH;
            size_t address_select,branch_output_address_select;
            if(!merge_node_vector[i]->branch_node){
                address_select  = merge_node_vector[i-1]->Have_Maxpooling ? merge_node_vector[i-1]->pool_address : merge_node_vector[i-1]->output_address;
                merge_node_vector[i]->input_address.push_back(address_select/*offset_vector[std::stoi(tmp)-2]*/);//this 2 is preset the count number
                if(merge_node_vector[i]->Concat_output_control && merge_node_vector[i]->Have_Maxpooling){
                    const size_t layer_output_ch             = std::ceil(merge_node_vector[i]->output_channel/(double)Depth_select)*Depth_select;
                    auto pool_address_calc = std::pow(merge_node_vector[i]->input_tile_number,2)*std::pow(merge_node_vector[i]->input_tile_size,2)*layer_output_ch*DATA_WIDTH;
                    merge_node_vector[i]->output_address     = merge_node_vector[i]->input_address[0] + ((merge_node_vector[i]->Have_Upsample) ? std::pow(merge_node_vector[i]->Upsample_size,2)*output_size : output_size);
                    merge_node_vector[i]->pool_address       = merge_node_vector[i]->output_address + ((merge_node_vector[i]->Have_Upsample) ? std::pow(merge_node_vector[i]->Upsample_size,2)*pool_address_calc : pool_address_calc);
                }
                else if(merge_node_vector[i]->Have_Maxpooling){
                    merge_node_vector[i]->pool_address       = merge_node_vector[i]->input_address[0] + ((merge_node_vector[i]->Have_Upsample) ? std::pow(merge_node_vector[i]->Upsample_size,2)*output_size : output_size); //這邊我要說的是，我前面推完output_address但是我要做concat，那我就會把資料丟在我之前做output_address後面再去加上一次因為我要丟兩個地方
                }
                else if(!merge_node_vector[i]->Have_Maxpooling){
                    merge_node_vector[i]->output_address     = merge_node_vector[i]->input_address[0] + ((merge_node_vector[i]->Have_Upsample) ? std::pow(merge_node_vector[i]->Upsample_size,2)*output_size : output_size);
                }else{
                    std::cerr<<"sorry we dont have this node"<<std::endl;
                }
                merge_node_vector[i]->Previous_node_OCH.push_back( merge_node_vector[i-1]->output_channel);
            } 
            else{
                size_t branch_node_jump_location_select;
                branch_node_jump_location_select = branch_node_jump_location_inside[branch_node_jump_location_inside.size()-1];
                address_select  = merge_node_vector[branch_node_jump_location_select]->output_address;
                std::cout<<"branch_node "<<branch_node_jump_location_select<<" "<<address_select<<std::endl;
                branch_output_address_select  = merge_node_vector[i-1]->Have_Maxpooling ? merge_node_vector[i-1]->pool_address : merge_node_vector[i-1]->output_address;//要這樣做的原因是因為我們現在輸入要抓到之前的輸出位子，但是我們的輸出位置會因為我們再做節點跳躍，所以要是在最後面輸出的位置在去加上資料才對
                merge_node_vector[i]->input_address.push_back(address_select/*offset_vector[std::stoi(tmp)-2]*/);//this 2 is preset the count number
                if(merge_node_vector[i]->Concat_output_control && merge_node_vector[i]->Have_Maxpooling){
                    const size_t layer_output_ch                = std::ceil(merge_node_vector[i]->output_channel/(double)Depth_select)*Depth_select;
                    auto pool_address_calc = std::pow(merge_node_vector[i]->input_tile_number,2)*std::pow(merge_node_vector[i]->input_tile_size,2)*layer_output_ch*DATA_WIDTH;
                    merge_node_vector[i]->output_address     = branch_output_address_select + ((merge_node_vector[i]->Have_Upsample) ? std::pow(merge_node_vector[i]->Upsample_size,2)*output_size : output_size);
                    merge_node_vector[i]->pool_address       = branch_output_address_select + ((merge_node_vector[i]->Have_Upsample) ? std::pow(merge_node_vector[i]->Upsample_size,2)*pool_address_calc : pool_address_calc);
                }
                else if(merge_node_vector[i]->Have_Maxpooling){
                    merge_node_vector[i]->pool_address       = branch_output_address_select + ((merge_node_vector[i]->Have_Upsample) ? std::pow(merge_node_vector[i]->Upsample_size,2)*output_size : output_size); //這邊我要說的是，我前面推完output_address但是我要做concat，那我就會把資料丟在我之前做output_address後面再去加上一次因為我要丟兩個地方
                }
                else if(!merge_node_vector[i]->Have_Maxpooling){
                    merge_node_vector[i]->output_address     = branch_output_address_select + ((merge_node_vector[i]->Have_Upsample) ? std::pow(merge_node_vector[i]->Upsample_size,2)*output_size : output_size); 
                }else{
                    std::cerr<<"sorry we dont have this node"<<std::endl;
                }
                branch_node_jump_location_inside.pop_back();
                merge_node_vector[i]->Previous_node_OCH.push_back( merge_node_vector[branch_node_jump_location_select]->output_channel);
            }
            offset_vector.push_back(merge_node_vector[i]->output_address);
        }
    }
}
size_t ten_valid_type(std::shared_ptr<pstruct> &merge_node,size_t &i,size_t &j){
    auto tile_number = merge_node->input_tile_number;
    size_t value;
    if(tile_number==1&&merge_node->input_padding_size!=0)value = 9;
    else if(tile_number==1&&merge_node->input_padding_size==0)value = 10;
	else if(i==0&&j==0)value = 0;
	else if(i==0 && j!=tile_number-1)value = 1;
	else if(i==0 && j==tile_number-1)value = 2;
	else if(i!=0 && i<tile_number-1 && j==0)value = 3;
	else if(i!=0 && i<tile_number-1 && j!=tile_number-1)value = 4;
	else if(i!=0 && i<tile_number-1 && i!=tile_number-1 && j==tile_number-1)value = 5;
	else if(i==tile_number-1 && j==0)value = 6;
	else if(i==tile_number-1 && j!=tile_number-1)value = 7;
	else value = 8;
    return value;
}
void pstruct::gen_out_addr(std::shared_ptr<pstruct> &merge_node){
    const size_t i_tile_num = merge_node->input_tile_number;
    const size_t next_tile_size = merge_node->next_tile_size;
    const size_t next_tile_num = merge_node->output_tile_number;
    const size_t o_tile_count = i_tile_num / next_tile_num;
    const size_t o_tile_size = merge_node->output_tile_size;
    const size_t next_input_tile_num = merge_node->output_tile_number;
    const size_t input_tile_size = merge_node->input_tile_size;
    const size_t Depth_select = merge_node->class_name=="Dense" ?  DATA_Depth_24 : merge_node->kernel_size==1 ? DATA_Depth_24 : DATA_DEPTH;//這樣的寫法代表說，如果今天是CONV那他會判斷要8or24，如果今天是dense那也不用怕因為是-1他會直接給我們24
    const size_t two_byte_per_ch = Depth_select*2;
    const size_t channel_align = std::ceil(merge_node->output_channel/(double)Depth_select);
    const size_t branch_input_tile_size = merge_node->branch_input_tile_size;
    std::cout<<"class_name          "<<merge_node->class_name<<std::endl;
    std::cout<<"i_tile_num          "<<i_tile_num<<std::endl;
    std::cout<<"next_tile_size      "<<next_tile_size<<std::endl;
    std::cout<<"next_tile_num       "<<next_tile_num<<std::endl;
    std::cout<<"o_tile_count        "<<o_tile_count<<std::endl;
    std::cout<<"o_tile_size         "<<o_tile_size<<std::endl;
    std::cout<<"next_input_tile_num "<<next_input_tile_num<<std::endl;
    std::cout<<"channel_align       "<<channel_align<<std::endl;
    std::cout<<"output_address      "<<merge_node->output_address<<std::endl;
    std::cout<<"Depth_select        "<<Depth_select<<std::endl;
    std::cout<<std::endl;
    if(merge_node->class_name!="Dense" && merge_node->class_name!="Flatten"){
        merge_node->out_addr_set.resize(i_tile_num*i_tile_num*channel_align,0);
        merge_node->out_addr_set_pool.resize(i_tile_num*i_tile_num*channel_align,0);
        merge_node->valid_address_vec.resize(i_tile_num*i_tile_num*channel_align,0);
        size_t tmp,tmp_pool;
        for(size_t i = 0; i < i_tile_num; i++){//FIX
            if(merge_node->Have_Maxpooling && merge_node->Concat_output_control){
                tmp      =  merge_node->output_address + (i/(i_tile_num)) * branch_input_tile_size * branch_input_tile_size * i_tile_num * two_byte_per_ch;//8ch per data 2 byte
                tmp      += (i % i_tile_num) * branch_input_tile_size * branch_input_tile_size * i_tile_num * two_byte_per_ch; //add one tiles next_input_tile_number because you got the same line;
                tmp_pool =  merge_node->pool_address + (i/(o_tile_count)) * next_tile_size * next_tile_size * next_input_tile_num * two_byte_per_ch;//8ch per data 2 byte
                tmp_pool += (i % o_tile_count) * o_tile_size * next_tile_size * next_input_tile_num * two_byte_per_ch; //add one tiles next_input_tile_number because you got the same line;
            }
            else if(merge_node->Have_Maxpooling){
                tmp_pool =  merge_node->pool_address + (i/(o_tile_count)) * next_tile_size * next_tile_size * next_input_tile_num * two_byte_per_ch;//8ch per data 2 byte
                tmp_pool += (i % o_tile_count) * o_tile_size * next_tile_size * next_input_tile_num * two_byte_per_ch; //add one tiles next_input_tile_number because you got the same line;
            }
            else if(!merge_node->Have_Maxpooling){
                tmp      =  merge_node->output_address + (i/(o_tile_count)) * next_tile_size * next_tile_size * next_input_tile_num * two_byte_per_ch;//8ch per data 2 byte
                tmp      += (i % o_tile_count) * o_tile_size * next_tile_size * next_input_tile_num * two_byte_per_ch; //add one tiles next_input_tile_number because you got the same line;
            }
            for(size_t j = 0; j < i_tile_num; j++){
                for(size_t och = 0; och < channel_align; och++){
                    size_t index = (i*i_tile_num+j)*channel_align+och;
                    if(merge_node->Have_Maxpooling){
                        merge_node->out_addr_set_pool[index] = tmp_pool;
                        merge_node->out_addr_set_pool[index] += (j/o_tile_count) * next_tile_size *  two_byte_per_ch; //sub one times next_tile_size because you get the same line;
                        merge_node->out_addr_set_pool[index] += (j%o_tile_count) * o_tile_size *  two_byte_per_ch;
                        merge_node->out_addr_set_pool[index] += och * next_tile_size * next_tile_size * next_input_tile_num * next_input_tile_num * two_byte_per_ch;
                    }
                    if(merge_node->Concat_output_control || !merge_node->Have_Maxpooling){
                        merge_node->out_addr_set[index] = tmp;
                        merge_node->out_addr_set[index] += (j/i_tile_num) *  branch_input_tile_size *  two_byte_per_ch; //sub one times next_tile_size because you get the same line;
                        merge_node->out_addr_set[index] += (j%i_tile_num) * branch_input_tile_size *  two_byte_per_ch;
                        merge_node->out_addr_set[index] += och * branch_input_tile_size * branch_input_tile_size * i_tile_num * i_tile_num * two_byte_per_ch;
                    }
                    auto tile_type = ten_valid_type(merge_node,i,j);
                    //auto input_address = merge_node->input_address[0];
                    auto input_address = 0;
                    //auto input_tile_size = (!merge_node->input_padding_size)? merge_node->input_tile_size : merge_node->input_tile_size + merge_node->input_padding_size*2;
                    auto input_tile_size = merge_node->input_tile_size;
                    auto input_tile_number = merge_node->input_tile_number;
                    if(tile_type==0 || tile_type==9 || tile_type==10){
                        //merge_node->valid_address_vec[index] = (!merge_node->input_padding_size) ? input_address : input_address + (input_tile_size*input_tile_number+1)*BUS_WIDTH;
                        merge_node->valid_address_vec[index] = input_address;
                    }else if(tile_type==1 || tile_type==2){
                        // merge_node->valid_address_vec[index] = (!merge_node->input_padding_size) ? input_address + ((input_tile_size*j)-1)*BUS_WIDTH 
                                                                                                //  : input_address + (input_tile_size*input_tile_number+(j*input_tile_size)-1)*BUS_WIDTH;
                        merge_node->valid_address_vec[index] = input_address + ((input_tile_size*j)-1)*two_byte_per_ch ;
                    }else if(tile_type==3 || tile_type==6){
                        // merge_node->valid_address_vec[index] = (!merge_node->input_padding_size) ? input_address + std::pow(input_tile_size,2)*input_tile_number*(i-1)*BUS_WIDTH + input_tile_size*(input_tile_size-1)*input_tile_number*BUS_WIDTH
                                                                                                //  : input_address + (input_tile_size*(input_tile_size-1)*input_tile_number+1)*BUS_WIDTH;
                        merge_node->valid_address_vec[index] = input_address + std::pow(input_tile_size,2)*input_tile_number*(i-1)*two_byte_per_ch + input_tile_size*(input_tile_size-1)*input_tile_number*two_byte_per_ch;
                    }else if(tile_type==4 || tile_type==5 || tile_type== 7 || tile_type==8){
                        merge_node->valid_address_vec[index] = input_address + ((i-1)*std::pow(input_tile_size,2)*input_tile_number + input_tile_size*(input_tile_size-1)*input_tile_number + (input_tile_size*j)-1)*two_byte_per_ch;
                    }else{
                        std::invalid_argument("sorry we cant support this type");
                    }
                    //std::cout<<tile_type<<" "<<merge_node->valid_address_vec[index]<<std::endl;
                }
            }
        }
    }else if(merge_node->class_name=="Dense"){ // i_tile_num and next_tile_num in dense 都是看 f_ix f_iy_fich ，舉例來說這邊 i_tile_num會等於 4*4*上高斯(16/24) ，next_tile_num會等於160(och)但是應該看成三圍 24*7*1=168
        size_t FC_input_layer_num,FC_input_layer_x_mul_y,FC_input_layer_RD,FC_output_layer_num;
        size_t Dense_flat_size = 1;//Dense 平面的大小我這邊都設定為一，讓他用深度去走完，但迴圈我還是寫三層方便識別
        size_t tmp,index;
        size_t input_address = 0;
        merge_node->out_addr_set.resize(round_ch_24(i_tile_num)*round_ch_24(next_tile_num),0);
        merge_node->valid_address_vec.resize(round_ch_24(i_tile_num)*round_ch_24(next_tile_num),0);
        std::cout<<"little func!!!! "<<merge_node->out_addr_set.size()<<" "<<merge_node->valid_address_vec.size()<<std::endl;
        if(!merge_node->IF_PRE_NODE_IS_DENSE){
            //FC_input_layer_num = merge_node->Dense_size_input_x * merge_node->Dense_size_input_y * round_ch_24(merge_node->input_channel);
            FC_input_layer_x_mul_y = merge_node->Dense_size_input_x * merge_node->Dense_size_input_y;
            FC_input_layer_RD = round_ch_24(merge_node->input_channel);
            FC_output_layer_num = round_ch_24(merge_node->Dense_output_node);
            for(int i=0;i<Dense_flat_size;i++){
                for(int j=0;j<FC_input_layer_RD;j++){
                    for(int x=0;x<FC_input_layer_x_mul_y;x++){
                        for(int z=0;z<FC_output_layer_num;z++){
                            index = (j*FC_input_layer_x_mul_y*FC_output_layer_num) + (x*FC_output_layer_num) + z;
                            merge_node->out_addr_set[index] = merge_node->output_address;
                            merge_node->valid_address_vec[index] = input_address + j*FC_input_layer_x_mul_y*DATA_Depth_24 * 2;//2byte
                        }
                    }
                }
            }
        }
        else{
            FC_input_layer_num = round_ch_24(merge_node->Dense_input_node);
            FC_output_layer_num = round_ch_24(merge_node->Dense_output_node);
            for(int i=0;i<Dense_flat_size;i++){
                for(int j=0;j<FC_input_layer_num;j++){
                    for(int z=0;z<FC_output_layer_num;z++){
                        index = j*FC_output_layer_num+z;
                        merge_node->out_addr_set[index] = merge_node->output_address;
                        merge_node->valid_address_vec[index] = input_address; //因為前面已經有一個DENSE，所以後面可以都看成是一個點很多深度input_address都是一樣的
                    }
                }
            }
        }
    }
}
void pstruct::gen_t_type_m(std::shared_ptr<pstruct> &merge_node){
    std::bitset<4> ten_type;
    merge_node->tile_padding_type.reserve(merge_node->out_addr_set.size());
    if(merge_node->class_name=="Dense"){
        merge_node->tile_padding_type.resize(merge_node->out_addr_set.size(),10);
        return;
    }
    const size_t i_tile_num = merge_node->input_tile_number;
    const size_t next_tile_num = merge_node->output_tile_number;
    const size_t o_tile_count = i_tile_num / next_tile_num;
    for(size_t i = 0;i<i_tile_num;i++){
        for(size_t j = 0;j<i_tile_num;j++){
            ten_type = 0b0000;
            if(merge_node->input_tile_number==1&&merge_node->input_padding_size!=0)ten_type = 9;
            else if(merge_node->input_tile_number==1&&merge_node->input_padding_size==0)ten_type = 10;
            else if(i==0 && j==0){
                ten_type = 0;
            } else if(i==0 && j!=i_tile_num-1){
                ten_type = 1;
            } else if(i==0 && j==i_tile_num-1){
                ten_type = 2;
            } else if(i!=0 && i<i_tile_num-1 &&j==0){
                ten_type = 3;
            } else if(i!=0 && i<i_tile_num-1 &&j!=i_tile_num-1){
                ten_type = 4;
            } else if(i!=0 && i<i_tile_num-1 && i!=i_tile_num-1 && j==i_tile_num-1){
                ten_type = 5;
            } else if(i==i_tile_num-1 && j==0){
                ten_type = 6;
            } else if(i==i_tile_num-1 && j!=i_tile_num-1){
                ten_type = 7;
            } else if(i==i_tile_num-1 && j==i_tile_num-1){
                ten_type = 8;
            } else std::cerr<<"we dont have this type"<<std::endl;
            u_char tmp = ten_type.to_ulong();
            merge_node->tile_padding_type.push_back(tmp);
        }
    }
}
void pstruct::get_tile_info(std::vector<std::shared_ptr<pstruct>> &merge_node_vector){
    for(auto merge_node : merge_node_vector){
        gen_out_addr(merge_node);
        gen_t_type_m(merge_node);
        const size_t Depth_select = merge_node->class_name=="Dense" ?  DATA_Depth_24 : merge_node->kernel_size==1 ? DATA_Depth_24 : DATA_DEPTH;
        const size_t i_tile_num  = merge_node->input_tile_number;
        const size_t next_tile_num = merge_node->output_tile_number;
        if(merge_node->class_name!="Dense"){
            const size_t ICH_Round = std::ceil(merge_node->input_channel/(double)Depth_select);
            const size_t OCH_Round = std::ceil(merge_node->output_channel/(double)Depth_select);
            merge_node->Tile_Info_Number = i_tile_num * i_tile_num * ICH_Round * OCH_Round; 
            const size_t tile_info_num  = merge_node->Tile_Info_Number;
            merge_node->tile_info.resize(tile_info_num*8,0);
            const size_t input_tile_number = merge_node->input_tile_number; //4*4*16
            size_t tmp_in_addr = 0;
            size_t index = 0;
            size_t tile_num_count = 0;
            const size_t max_tile_num = std::floor(HW_INFO_MAX/(double)ICH_Round)*(ICH_Round);
            const size_t HW_KERNEL_SIZE = std::pow(merge_node->kernel_size,2);
            for(size_t i=0;i<input_tile_number*input_tile_number;i++){//x y z input_feature_size input_feature_size och_rd 
                for(size_t j=0;j<OCH_Round;j++){//conv ok flatten fc_x fc_y och_rd but always 1
                    size_t input_addr_count = 0;
                    size_t ich_count = 0;
                    for(const size_t layer_ich : merge_node->Previous_node_OCH){
                        //std::cout<<"layer_ich:"<<layer_ich<<std::endl;
                        for(size_t k = 0;k<std::ceil(layer_ich/(double)Depth_select);k++){
                            index = ((i * OCH_Round) +j) * ICH_Round + ich_count;
                            merge_node->tile_info[index*8] = merge_node->out_addr_set[i*OCH_Round+j];
                            merge_node->tile_info[index*8+1] = (j * ICH_Round + ich_count) * (HW_KERNEL_SIZE * Depth_select * Depth_select * DATA_WIDTH) + merge_node->weight_address;
                            merge_node->tile_info[index*8+1] += (j * Depth_select * 2 * DATA_WIDTH);
                            tmp_in_addr = merge_node->valid_address_vec[i*OCH_Round+j] + merge_node->input_address[input_addr_count];
                            tmp_in_addr += k * std::pow(merge_node->input_tile_number,2) * std::pow(merge_node->input_tile_size,2) * Depth_select * 2;
                            merge_node->tile_info[index*8+2] = tmp_in_addr;
                            merge_node->tile_info[index*8+3] = merge_node->out_addr_set_pool[i*OCH_Round+j];//pooling_addr fix the address if the tile need to concat #TODO
                            if(ich_count == ICH_Round-1) //Is_Last_CHannel
                                merge_node->tile_info[index*8+4] = 1;
                            else
                                merge_node->tile_info[index*8+4] = 0;
                            if(ich_count !=0)//Have_A_ccumulate
                                merge_node->tile_info[index*8+4] |= 1<<1;
                            else
                                merge_node->tile_info[index*8+4] |= 0<<1;
                            if((i == input_tile_number*input_tile_number-1) && (j == OCH_Round-1) && (ich_count == ICH_Round-1))//Is_Final_tile
                                merge_node->tile_info[index*8+4] |= 1 << 2;
                            else
                                merge_node->tile_info[index*8+4] |= 0 << 2;
                            if((tile_num_count%max_tile_num) == max_tile_num-1){
                                merge_node->tile_info[index*8+4] |= 1 << 2;
                            }
                            merge_node->tile_info[index*8+4] |= (merge_node->tile_padding_type[i]&0xf) << 3;
                            merge_node->tile_info[index*8+4] |= (merge_node->input_tile_size&0b111111)<<7;         //input_tile_size_row
                            merge_node->tile_info[index*8+4] |= (merge_node->input_tile_size&0b111111)<<13;        //input_tile_size_col
                            //merge_node->tile_info[index*8+4] |= (weight_len&0b1111111)<<19;       //weight_len 74
                            if(ich_count == ICH_Round-1 && merge_node->Have_BatchNormalization || merge_node->Have_bias)
                                merge_node->tile_info[index*8+4] |= (merge_node->kernel_size==3) ? (weight_len_last_3&0b1111111)<<19 : (weight_len_last_1&0b1111111)<<19;       //weight_len 74
                            else
                                merge_node->tile_info[index*8+4] |= (weight_len_not_last&0b1111111)<<19;       //weight_len 74
                            if(ich_count == ICH_Round-1 && merge_node->Have_Maxpooling) //Is_Last_CHannel && have max_pooling
                                merge_node->tile_info[index*8+4] |= 1<<26;
                            else
                                merge_node->tile_info[index*8+4] |= 0<<26; 
                            if(merge_node->kernel_size==1){
                                merge_node->tile_info[index*8+4] |= (ich_count==ICH_Round-1) ? (int(std::ceil((merge_node->input_channel%Depth_select)/(double)DATA_DEPTH))&0b11)<<27 : (3&0b11)<<27;
                                merge_node->tile_info[index*8+4] |= (j==OCH_Round-1) ?         (int(std::ceil((merge_node->output_channel%Depth_select)/(double)DATA_DEPTH))&0b11)<<29 : (3&0b11)<<29;
                            }else{
                                merge_node->tile_info[index*8+4] |= (3&0b11)<<27;
                                merge_node->tile_info[index*8+4] |= (3&0b11)<<29;
                            }
                            size_t Output_tile_size_with_out_pooling   =  merge_node->input_tile_size / (merge_node->kernel_stride!=0 ? merge_node->kernel_stride : 1); 
                            merge_node->tile_info[index*8+4] |= (Output_tile_size_with_out_pooling&0b000001)<<31;          //output_tile_size_row
                            merge_node->tile_info[index*8+5]  = (Output_tile_size_with_out_pooling&0b111110)>>1;           //output_tile_size_row
                            merge_node->tile_info[index*8+5] |= (Output_tile_size_with_out_pooling&0b111111)<<5;           //output_tile_size_col
                            //5bit control
                            merge_node->tile_info[index*8+7]  = 1<<12;                          //Reload
                            merge_node->tile_info[index*8+7] |= (index%2==1) ? 0<<16 : 1<<16;   //weight_buf_sel
                            merge_node->tile_info[index*8+7] |= (index%2==1) ? 0<<20 : 1<<20;   //input_buf_sel
                            merge_node->tile_info[index*8+7] |= 1<<24;                          //weight_loading
                            merge_node->tile_info[index*8+7] |= 1<<28;                          //input_loading
                            ich_count++;
                            tile_num_count++;
                        }
                        //std::cout<<"input_addr_count:"<<input_addr_count<<std::endl;
                        input_addr_count++;
                    }
                }
            }
        }else if(merge_node->class_name=="Dense"){
            size_t FC_input_layer_num,FC_input_layer_x_mul_y,FC_input_layer_RD,FC_output_layer_num;
            size_t Dense_flat_size = 1;//Dense 平面的大小我這邊都設定為一，讓他用深度去走完，但迴圈我還是寫三層方便識別
            size_t tmp,index;
            size_t input_address = 0;
            size_t change_number = 0;
            merge_node->tile_info.resize(round_ch_24(i_tile_num)*round_ch_24(next_tile_num)*8,0);
            std::cout<<"merge_node->tile_info.size() : "<<merge_node->tile_info.size()<<std::endl;
            if(!merge_node->IF_PRE_NODE_IS_DENSE){
                merge_node->Tile_Info_Number = merge_node->Dense_size_input_x * merge_node->Dense_size_input_y * round_ch_24(merge_node->input_channel) * merge_node->Dense_size_output_x * merge_node->Dense_size_output_y;
                FC_input_layer_x_mul_y = merge_node->Dense_size_input_x * merge_node->Dense_size_input_y;
                FC_input_layer_RD = round_ch_24(merge_node->input_channel);
                FC_output_layer_num = round_ch_24(merge_node->Dense_output_node);
                for(int i=0;i<Dense_flat_size;i++){
                    for(int j=0;j<FC_input_layer_RD;j++){
                        for(int x=0;x<FC_input_layer_x_mul_y;x++){
                            for(int z=0;z<FC_output_layer_num;z++){
                                index = (j*FC_input_layer_x_mul_y*FC_output_layer_num) + (x*FC_output_layer_num) + z;
                                merge_node->tile_info[index*8]   = merge_node->out_addr_set[index];
                                merge_node->tile_info[index*8+1] = merge_node->weight_address + index * DATA_Depth_24 * DATA_Depth_24 * DATA_WIDTH;
                                //std::cout<<merge_node->tile_info[index*8+1]<<std::endl;
                                if(j==FC_input_layer_RD-1 && x==FC_input_layer_x_mul_y-1){
                                    merge_node->tile_info[index*8+1] += (z) * DATA_Depth_24 * 2 * DATA_WIDTH;
                                }
                                //merge_node->tile_info[index*8+1] += (j*FC_input_layer_x_mul_y+x) * 1 * DATA_WIDTH; //bias
                                merge_node->tile_info[index*8+2] = (index%2==0) ? merge_node->valid_address_vec[index] + merge_node->input_address[0] 
                                                                                : merge_node->valid_address_vec[index] + merge_node->input_address[0] +  FC_input_layer_x_mul_y*DATA_Depth_24*DATA_WIDTH;//因為只會有一個進入點這邊是做DENSE
                                merge_node->tile_info[index*8+2] += j * FC_input_layer_x_mul_y*DATA_Depth_24*DATA_WIDTH;
                                merge_node->tile_info[index*8+3] = 0;//全連接不會有輸出在pooling的資料
                                if(j==FC_input_layer_RD-1 && x==FC_input_layer_x_mul_y-1)//Is_Last_CHannel
                                    merge_node->tile_info[index*8+4] = 1;
                                else
                                    merge_node->tile_info[index*8+4] = 0;
                                if(j!=0 || x!=0)//Have_A_ccumulate
                                    merge_node->tile_info[index*8+4] |= 1<<1;
                                else
                                    merge_node->tile_info[index*8+4] |= 0<<1;
                                if(j==FC_input_layer_RD-1 && x==FC_input_layer_x_mul_y-1 && z==FC_output_layer_num-1)//Is_Final_tile
                                    merge_node->tile_info[index*8+4] |= 1<<2;
                                else
                                    merge_node->tile_info[index*8+4] |= 0<<2;
                                merge_node->tile_info[index*8+4] |= (merge_node->tile_padding_type[i]&0xf) << 3;
                                merge_node->tile_info[index*8+4] |= (merge_node->Dense_size_input_x&0b111111)<<7;         //input_tile_size_row
                                merge_node->tile_info[index*8+4] |= (merge_node->Dense_size_input_y&0b111111)<<13;        //input_tile_size_col
                                if(j==FC_input_layer_RD-1 && x==FC_input_layer_x_mul_y-1)
                                    merge_node->tile_info[index*8+4] |= (merge_node->kernel_size==3) ? (weight_len_last_3&0b1111111)<<19 : (weight_len_last_1&0b1111111)<<19;       //weight_len 74
                                else
                                    merge_node->tile_info[index*8+4] |= (weight_len_not_last&0b1111111)<<19;       //weight_len 72
                                merge_node->tile_info[index*8+4] |= 0<<26;
                                //ICP OCP //要記得修改
                                merge_node->tile_info[index*8+4] |= (j==FC_input_layer_RD-1)  ? (int(std::ceil((merge_node->input_channel%Depth_select)/(double)DATA_DEPTH))&0b11)<<27   : (3&0b11)<<27; //要拿原始的資料像是 256 160這樣不會是拿384 168這樣
                                merge_node->tile_info[index*8+4] |= (3&0b11)<<29;
                                merge_node->tile_info[index*8+4] |= (merge_node->Dense_size_output_x&0b000001)<<31;          //output_tile_size_row
                                merge_node->tile_info[index*8+5]  = (merge_node->Dense_size_output_x&0b111110)>>1;           //output_tile_size_row
                                merge_node->tile_info[index*8+5] |= (merge_node->Dense_size_output_y&0b111111)<<5;           //output_tile_size_col
                                //5 bit control
                                if(FC_output_layer_num%2==0){ //AB為雙數
                                    //RELOAD
                                    if(index < 2){
                                        merge_node->tile_info[index*8+7] = 1<<12;
                                    }else if(j%2==1 && x==0 && z==0){
                                        merge_node->tile_info[index*8+7] = 1<<12;
                                    }else if(j%2==0 && x==0 && z==1){
                                        merge_node->tile_info[index*8+7] = 1<<12;
                                    }else{
                                        merge_node->tile_info[index*8+7] = 0<<12;
                                    }
                                    //weight_buf_sel
                                    merge_node->tile_info[index*8+7]     |= (index%2==1) ? 0<<16 : 1<<16;
                                    //input_buf_sel
                                    if(j%2==0){
                                        merge_node->tile_info[index*8+7] |= 1<<20;
                                    }else{
                                        merge_node->tile_info[index*8+7] |= 0<<20;
                                    }
                                    //weight_loading
                                    merge_node->tile_info[index*8+7]     |= 1<<24;
                                    //input_loading
                                    if(index < 2){
                                        merge_node->tile_info[index*8+7] |= 1<<28;
                                    }else if(j%2==1 && x==0 && z==0){
                                        merge_node->tile_info[index*8+7] |= 1<<28;
                                    }else if(j%2==0 && x==0 && z==1){
                                        merge_node->tile_info[index*8+7] |= 1<<28;
                                    }else{
                                        merge_node->tile_info[index*8+7] |= 0<<28;
                                    }
                                }else{//AB為單數
                                    //RELOAD
                                    if(index < 2){
                                        merge_node->tile_info[index*8+7] = 1<<12;
                                    }else if(x==0 && z==1){
                                        merge_node->tile_info[index*8+7] = 1<<12;
                                    }else{
                                        merge_node->tile_info[index*8+7] = 0<<12;
                                    }
                                    //weight_buf_sel
                                    merge_node->tile_info[index*8+7]     |= (index%2==1) ? 0<<16 : 1<<16;
                                    //input_buf_sel
                                    if(j%2==0){
                                        merge_node->tile_info[index*8+7] |= 1<<20;
                                    }else{
                                        merge_node->tile_info[index*8+7] |= 0<<20;
                                    }
                                    //weight_loading
                                    merge_node->tile_info[index*8+7] |= 1<<24;
                                    //input_loading
                                    if(index < 2){
                                        merge_node->tile_info[index*8+7] |= 1<<28;
                                    }else if(x==0 && z==1){
                                        merge_node->tile_info[index*8+7] |= 1<<28;
                                    }else{
                                        merge_node->tile_info[index*8+7] |= 0<<28;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            else{
                merge_node->Tile_Info_Number = merge_node->Dense_size_input_x*merge_node->Dense_size_input_y * merge_node->Dense_size_output_x * merge_node->Dense_size_output_y;
                FC_input_layer_num = round_ch_24(merge_node->Dense_input_node);
                FC_output_layer_num = round_ch_24(merge_node->Dense_output_node);
                for(int i=0;i<Dense_flat_size;i++){
                    for(int j=0;j<FC_input_layer_num;j++){
                        for(int z=0;z<FC_output_layer_num;z++){
                            index = j * FC_output_layer_num + z;
                            merge_node->tile_info[index*8] = merge_node->out_addr_set[index];
                            merge_node->tile_info[index*8+1] = merge_node->weight_address + index * DATA_Depth_24 * DATA_Depth_24 * DATA_WIDTH;
                            //std::cout<<merge_node->tile_info[index*8+1]<<std::endl;
                            if(j==FC_input_layer_num-1){
                                merge_node->tile_info[index*8+1] += (z) * DATA_Depth_24 * 2 * DATA_WIDTH;
                            }
                            //merge_node->tile_info[index*8+1] += j * 2 * DATA_WIDTH;//bias
                            merge_node->tile_info[index*8+2] = merge_node->valid_address_vec[index] + merge_node->input_address[0];
                            merge_node->tile_info[index*8+3] = 0;
                            if(j==FC_input_layer_num-1)//Is_Last_Channel 告訴硬體說這邊是最後一個channel要記得加上alpga bias
                                merge_node->tile_info[index*8+4] = 1;
                            else
                                merge_node->tile_info[index*8+4] = 0;
                            if(j!=0)//Have_A_ccumulate
                                merge_node->tile_info[index*8+4] |= 1<<1;
                            else
                                merge_node->tile_info[index*8+4] |= 0<<1;
                            if(j==FC_input_layer_num-1 && z==FC_output_layer_num-1)//Is_Final_tile 告訴硬體說我這邊要輸出拉因為已經是最後一個了
                                merge_node->tile_info[index*8+4] |= 1<<2;
                            else
                                merge_node->tile_info[index*8+4] |= 0<<2;
                            merge_node->tile_info[index*8+4] |= (merge_node->tile_padding_type[i]&0xf) << 3;
                            merge_node->tile_info[index*8+4] |= (merge_node->Dense_size_input_x&0b111111)<<7;         //input_tile_size_row
                            merge_node->tile_info[index*8+4] |= (merge_node->Dense_size_input_y&0b111111)<<13;        //input_tile_size_col
                            if(j==FC_input_layer_num-1)
                                merge_node->tile_info[index*8+4] |= (merge_node->kernel_size==3) ? (weight_len_last_3&0b1111111)<<19 : (weight_len_last_1&0b1111111)<<19;       //weight_len 74
                            else
                                merge_node->tile_info[index*8+4] |= (weight_len_not_last&0b1111111)<<19;       //weight_len 74
                            merge_node->tile_info[index*8+4] |= 0<<26;
                            merge_node->tile_info[index*8+4] |= (3&0b11)<<27; //ICP 這邊都會拿滿 因為前面是DEBSE
                            merge_node->tile_info[index*8+4] |= (3&0b11)<<29; //OCP 這邊都會拿滿 因為前面是DEBSE
                            merge_node->tile_info[index*8+4] |= (merge_node->Dense_size_output_x&0b000001)<<31;          //output_tile_size_row
                            merge_node->tile_info[index*8+5]  = (merge_node->Dense_size_output_x&0b111110)>>1;           //output_tile_size_row
                            merge_node->tile_info[index*8+5] |= (merge_node->Dense_size_output_y&0b111111)<<5;           //output_tile_size_col
                            //5 bit control
                            if(FC_output_layer_num%2==0){ //AB為雙數
                                //RELOAD
                                if(index < 2){
                                    merge_node->tile_info[index*8+7] = 1<<12;
                                }else{
                                    merge_node->tile_info[index*8+7] = 0<<12;
                                }
                                //weight_buf_sel
                                merge_node->tile_info[index*8+7]     |= (index%2==1) ? 0<<16 : 1<<16;
                                //input_buf_sel
                                if(i%2==0){
                                    merge_node->tile_info[index*8+7] |= 1<<20;
                                }else{
                                    merge_node->tile_info[index*8+7] |= 0<<20;
                                }
                                //weight_loading
                                merge_node->tile_info[index*8+7]     |= 1<<24;
                                //input_loading
                                if(index < 2){
                                    merge_node->tile_info[index*8+7] |= 1<<28;
                                }else{
                                    merge_node->tile_info[index*8+7] |= 0<<28;
                                }
                            }else{//AB為單數
                                //RELOAD
                                if(index < 2){
                                    merge_node->tile_info[index*8+7] = 1<<12;
                                }else{
                                    merge_node->tile_info[index*8+7] = 0<<12;
                                }
                                //weight_buf_sel
                                merge_node->tile_info[index*8+7]     |= (index%2==1) ? 0<<16 : 1<<16;
                                //input_buf_sel
                                if(i%2==0){
                                    merge_node->tile_info[index*8+7] |= 1<<20;
                                }else{
                                    merge_node->tile_info[index*8+7] |= 0<<20;
                                }
                                //weight_loading
                                merge_node->tile_info[index*8+7]     |= 1<<24;
                                //input_loading
                                if(index < 2){
                                    merge_node->tile_info[index*8+7] |= 1<<28;
                                }else{
                                    merge_node->tile_info[index*8+7] |= 0<<28;
                                }
                            }
                        }
                    }
                }
            }
        }else{
            throw std::invalid_argument("this node is broke in func get_tile_info please check !!!!!!!!");
            //U CAN ADD FUNC IN THERE
        }
    }
}
std::vector<uint32_t> pstruct::Gen_Layer_Info(std::shared_ptr<pstruct> &merge_node){
    merge_node->Leaky_ReLU_alpha_FP = merge_node->Leaky_ReLU_alpha*std::pow(2, merge_node->quant_batch);
    std::vector<uint32_t>Inst(5,0);
    Inst[0] = merge_node->Is_LeakyReLU;
    Inst[0] |= (merge_node->Have_ReLU&1) << 1;
    Inst[0] |= ((merge_node->Have_BatchNormalization||merge_node->Have_bias)&1)<<2;
    //Inst[0] |= (merge_node->Have_Maxpooling&1)<<3;
    Inst[0] |= (merge_node->Batch_First&1)<<3;
    Inst[0] |= (merge_node->Bit_Serial&0b1111)<<4;
    Inst[0] |= (merge_node->pool_stride&0b11)<<8;
    Inst[0] |= (merge_node->pool_size&0b11)<<10;
    Inst[0] |= (merge_node->kernel_stride&0b11)<<12;
    Inst[0] |= (merge_node->kernel_size&0b11)<<14;
    //TODO quant_serias
    Inst[0] |= (merge_node->quant_batch_bias%0b111111)<<16;
    Inst[0] |= (merge_node->quant_finish&0b111111)<<22;
    //Inst[0] |= (merge_node->quant_batch&0b001111)<<28;
    //Inst[1] |= (merge_node->quant_batch&0b110000)>>2;
    Inst[0] |= (merge_node->quant_word_size&0b001111)<<28;
    Inst[1] |= (merge_node->quant_word_size&0b110000)>>4;
    Inst[1] |= (merge_node->quant_obuf&0b111111)<<2;
    //under condition with out the rule of pdf
    Inst[1] |= (merge_node->Tile_Info_Number&0xfff)<<8; //以第八層為例 ICH 256 OCH 64 = ceil(256/24)=11 ceil(64/24)=3 3*11=33條，但是因為我們硬體一次是128但是我們的Txt是256所以要乘以2來看=66又因為AXI R CH會少一 所以等於65
    Inst[1] |= (merge_node->Tile_Info_Addr&0x00000fff)<<20;
    Inst[2]  = (merge_node->Tile_Info_Addr&0xfffff000)>>12;
    //Inst[3] |= (merge_node->next_tile_size&0x3f) << 17;
    Inst[2] |= (merge_node->Have_Upsample&0x1) << 20;//TODO need to fix unsample 
    Inst[2] |= (merge_node->Leaky_ReLU_alpha_FP&0x07ff) << 21;
    Inst[3]  = (merge_node->Leaky_ReLU_alpha_FP&0xf800) >> 11;
    int32_t input_feature_offset_select = merge_node->kernel_size==1 ? std::pow(merge_node->input_feature_size,2) : merge_node->input_feature_size;//如果是3*3的話，要給予整個input_feature_size，但是如果是1*1那要給予這個tile的offset，例如這個tile是8*8那這個input_feature_offset就是64，最大限制是34*12
    int32_t input_feature_offset = merge_node->class_name=="Dense" ? (merge_node->Dense_size_input_x * merge_node->Dense_size_input_y) : input_feature_offset_select;
    Inst[3] |= (input_feature_offset&0xfff)<<5;
    int32_t output_feature_select = merge_node->Have_Maxpooling ? merge_node->output_feature_size*2 : merge_node->output_feature_size; //這邊幫忙回復到沒有做maxpooling的參數 但是裡面運算還是要有喔~~~
    int32_t output_feature_offset = merge_node->class_name=="Dense" ? (merge_node->Dense_size_output_x * merge_node->Dense_size_output_y) : merge_node->kernel_size==1 ? std::pow(output_feature_select,2) : output_feature_select;
    Inst[3] |= (output_feature_offset&0xfff)<<17;//input_feature_offset
    Inst[3] |= (merge_node->class_name=="Dense" ? 1&0b11 : 0&0b11)<<29; // u got four type 00 is conv_mode，01 is FC_mode，10 is DWS_mode，11 is DE_CONV_mode
    Inst[3] |= (merge_node->Concat_output_control&0b1)<<31;
    Inst[4] = merge_node->pooling_quant_finish&0b111111;
    return Inst;
}
std::vector<uint32_t> pstruct::gen_layer_info_data(std::vector<std::shared_ptr<pstruct>> &merge_node_vector){
    std::vector<uint32_t> layer_info_data;
    const size_t LAYER_INFO_DATA_SIZE = 5; //160bit //5 slave reg store
    size_t tmp_addr = LAYER_ADDR_OFFSET; //65536 or 0
    size_t tmp_layer_count = 0;
    bool first_tile_addr_op = false;
    
    for(auto &merge_node : merge_node_vector){
        tmp_layer_count += std::ceil(merge_node->Tile_Info_Number / (double)HW_INFO_MAX); //how many times that we can finish the calc 不可以大於1024因為這邊是我們硬體的使用限制，超過沒有空間可以放
    }
    //layer_info_data.reserve(merge_node_vector.size()*tmp_layer_count*LAYER_INFO_DATA_SIZE);//這邊我有做過小測試，當初只是希望先抓個代蓋例如我現在再測試左邊八層帶入公式就是8*8*5=320，但是基本上我們只需要8*5就夠了因為我們的資料還是一樣多
    layer_info_data.reserve(tmp_layer_count*LAYER_INFO_DATA_SIZE);//這邊我有做過小測試，當初只是希望先抓個代蓋例如我現在再測試左邊八層帶入公式就是8*8*5=320，但是基本上我們只需要8*5就夠了因為我們的資料還是一樣多
    size_t FOUR_K_BOUNDARY_ZCU_OR_ZEDBOARD_NUMBER = BUS_WIDTH==8 ? 4 : 2; //32/BUS_WIDTH
    for(auto &merge_node : merge_node_vector){
        const size_t Depth_select = merge_node->class_name=="Dense" ?  DATA_Depth_24 : merge_node->kernel_size==1 ? DATA_Depth_24 : DATA_DEPTH;
        const size_t tile_num = merge_node->Tile_Info_Number * FOUR_K_BOUNDARY_ZCU_OR_ZEDBOARD_NUMBER;
        std::cout<<"tile_info_number: "<<merge_node->Tile_Info_Number<<std::endl;
        size_t tile_num_tmp = 0;
        merge_node->Tile_Info_Addr += tmp_addr;
        std::cout<<"-------------------------pre_layer_info----------------------"<<std::endl;
        for(size_t j=0;j<tile_num;j+=tile_num_tmp){
            tile_num_tmp = tile_num-j;
            if(tile_num_tmp > HW_INFO_MAX*FOUR_K_BOUNDARY_ZCU_OR_ZEDBOARD_NUMBER){//我們這邊需要判斷的是我們每次的資料量多寡會不會跨到4K邊界，但是對於zedboard or zcu來說雖然都是一個tile，但是tile的資料量不一樣，zedboard 4in4out，zcu 8in8out所以才會看到zcu反而會比較少就需要控管4k邊界的問題，因為他量大所以一半就已經到邊界了
                tile_num_tmp = HW_INFO_MAX*FOUR_K_BOUNDARY_ZCU_OR_ZEDBOARD_NUMBER;
                tile_num_tmp = std::floor(tile_num_tmp/std::ceil(merge_node->input_channel/(double)Depth_select))*std::ceil(merge_node->input_channel/(double)Depth_select);
            }
            if(!first_tile_addr_op){
                first_tile_addr_op = true;
            }else{
                merge_node->Tile_Info_Addr = tmp_addr;
            }
            merge_node->Tile_Info_Number = tile_num_tmp -1;
            const std::vector<uint32_t>layer_info = merge_node->Gen_Layer_Info(merge_node);
            for(auto i : layer_info){
                std::cout<<i<<std::endl;
            }
            layer_info_data.insert(layer_info_data.end(),layer_info.begin(),layer_info.end());
            tmp_addr +=tile_num_tmp * BUS_WIDTH;
        }
    }
    return layer_info_data;
}
void pstruct::dump_layer_info_sim(const std::string &filename,const std::vector<uint32_t> &layer_info_data){
    const size_t LAYER_INFO_DATA_SIZE = 5;
    std::ofstream out(filename);
    if(!out.is_open()){
        std::cerr<<"File "<<filename<<" Can not open"<<std::endl;
        throw std::invalid_argument("Can not open file");
    }
    if(layer_info_data.size()%LAYER_INFO_DATA_SIZE!=0){
        throw std::invalid_argument("Layer Info format error");
    }
    for(size_t i=0;i<layer_info_data.size()/LAYER_INFO_DATA_SIZE;i++){
        for(int j = LAYER_INFO_DATA_SIZE-1;j>=0;j--){
            out << fmt::format("{:08x}",layer_info_data[j+i*LAYER_INFO_DATA_SIZE]);
            std::cout<<fmt::format("{:08x}",layer_info_data[j+i*LAYER_INFO_DATA_SIZE]);
        }
        out<< "\n";
        std::cout<<"\n";
    }
    out.close();
}
void pstruct::dump_layer_info_data_bin(const std::string &filename, const std::vector<uint32_t> &layer_info_data){
    const size_t data_size = size_in_byte(layer_info_data);
    std::cout<<data_size<<std::endl;
    int fd = open(filename.c_str(), O_CREAT | O_RDWR | O_SYNC, S_IRUSR | S_IWUSR);
    if(fd < 0){
        throw std::invalid_argument("dump_bin : Can not open dump bin file.");
    }
    int write_num = write(fd, layer_info_data.data(), data_size);
    close(fd);
    if(write_num < 0){
        throw std::invalid_argument("dump_bin : Can not write dump bin file.");
    }
}
void pstruct::dump_total_tile_sim(const std::string &filename,const std::vector<std::shared_ptr<pstruct>> &merge_node_vector){
    std::ofstream out(filename);
    std::vector<uint32_t> data;
    const size_t PRE_TILE_BYTE = 32;
    const size_t TILE_COUNT  = PRE_TILE_BYTE/(sizeof(uint32_t));
    if(!out.is_open()){
        std::cerr << "can not open "<< filename<< std::endl;
        throw std::invalid_argument("Can not open the file for dump tile");
    }
    for(auto merge_node : merge_node_vector){
        if(merge_node->tile_info.size() & TILE_COUNT !=0){
            throw std::invalid_argument("Tile Info format error");
        }
        //std::cout<<"Tile_count"<<TILE_COUNT<<std::endl;
        for(size_t i=0;i<(merge_node->tile_info.size()/TILE_COUNT);i++){
            for(int j = TILE_COUNT-1;j>=0;j--){
                if(j<4){
                    if(j==1){//weight_addr
                        out << fmt::format("{:08x}",merge_node->tile_info[i*TILE_COUNT+j]+WEIGHT_BASE_ADDR);
                        //if(merge_node->class_name=="Dense" && !merge_node->IF_PRE_NODE_IS_DENSE)
                        //    std::cout<<"look in here "<<merge_node->tile_info[i*TILE_COUNT+j]<<std::endl;
                    }
                    else if(j==0){
                        if(merge_node->Have_Maxpooling && merge_node->Concat_output_control)
                            out << fmt::format("{:08x}",merge_node->tile_info[i*TILE_COUNT+j]+DATA_BASE_ADDR);
                        else if(!merge_node->Have_Maxpooling)
                            out << fmt::format("{:08x}",merge_node->tile_info[i*TILE_COUNT+j]+DATA_BASE_ADDR);
                        else
                            out << fmt::format("{:08x}",merge_node->tile_info[i*TILE_COUNT+j]);
                    }
                    else if(j==3){
                        if(merge_node->Have_Maxpooling)
                            out << fmt::format("{:08x}",merge_node->tile_info[i*TILE_COUNT+j]+DATA_BASE_ADDR);
                        else
                            out << fmt::format("{:08x}",merge_node->tile_info[i*TILE_COUNT+j]);
                    }
                    else{
                        out << fmt::format("{:08x}",merge_node->tile_info[i*TILE_COUNT+j]+DATA_BASE_ADDR);
                    }
                }else{
                    out << fmt::format("{:08x}",merge_node->tile_info[i*TILE_COUNT+j]);
                }
            }
            out<<"\n";
        }
    }
}
void pstruct::dump_tile_bin(const std::string &filename, const std::vector<std::shared_ptr<pstruct>> &merge_node_vector){
    
    size_t total_tile_size = 0;
    std::vector<uint32_t> total_tile_tmp;

    for(auto &merge_node : merge_node_vector){
        total_tile_size += size_in_byte(merge_node->tile_info);
    }
    std::cout << total_tile_size << std::endl;
    total_tile_tmp.reserve(total_tile_size/(sizeof(uint32_t)));

    for(auto &merge_node : merge_node_vector){
        total_tile_tmp.insert(total_tile_tmp.end(), merge_node->tile_info.begin() , merge_node->tile_info.end());
    }
    
    int fd = open(filename.c_str(), O_CREAT | O_RDWR | O_SYNC, S_IRUSR | S_IWUSR);
    if(fd < 0){
        throw std::invalid_argument("dump_bin : Can not open dump bin file.");
    }
    int write_num = write(fd, total_tile_tmp.data(), total_tile_size);
    close(fd);
    if(write_num < 0){
        throw std::invalid_argument("dump_bin : Can not write dump bin file.");
    }
}
std::string weight_type(const size_t num,const size_t need_file_num){
    if(need_file_num==6){
        switch(num){
            case 0:
                return "weight 0";
            case 1:
                return "bias 1";
            case 2:
                return "beta 2";
            case 3:
                return "gamma 3";
            case 4:
                return "variance 4";
            case 5:
                return "mean 5";
            default:
                return "None";
        }
    }
    else if(need_file_num==5){
        switch(num){
            case 0:
                return "weight 0";
            case 1:
                return "beta 1";
            case 2:
                return "gamma 2";
            case 3:
                return "variance 3";
            case 4:
                return "mean 4";
            default:
                return "None";
        }
    }
    else if(need_file_num==2){
        switch(num){
            case 0:
                return "weight 0";
            case 1:
                return "bias 1";
            default:
                return "None";
        }
    }
    else if(need_file_num==1){
        switch(num){
            case 0:
                return "weight 0";
            default:
                return "None";
        }
    }
    else{
        throw std::invalid_argument("func weight_type error please check");
    }
}
std::vector<float>get_weight_from_file(std::ifstream &in,const size_t need_num){
    std::vector<float> data;
    data.reserve(need_num);
    float tmp;
    for(size_t i = 0;i<need_num;i++){
        in>>tmp;
        if(in.eof()){
            std::cout<<tmp<<std::endl;
            throw std::invalid_argument("Weight need number greater than weight file"); 
        }
        data.push_back(tmp);
    }
    in>>tmp;
    if(!in.eof()){
        throw std::invalid_argument("Weight need number less then weight file");
    }
    return data;
}
std::vector<short>pstruct::gen_layer_weight(const std::shared_ptr<pstruct> &merge_node,const std::vector<std::string> &path_set){
    std::vector<std::vector<float>> data_in;
    const size_t Depth_select = (merge_node->class_name=="Dense")   ? DATA_Depth_24 
                                                                    : merge_node->kernel_size==1 ? DATA_Depth_24 : DATA_DEPTH; //這邊要做選擇是哪個深度要做判定，1*1 or 3*3
    const size_t OCH_NUM = (merge_node->class_name!="Dense")        ? std::ceil(merge_node->output_channel/(double)Depth_select)
                                                                    : std::ceil(merge_node->Dense_output_channel/(double)Depth_select);
    const size_t ICH_NUM = (merge_node->class_name!="Dense")        ? std::ceil(merge_node->input_channel/(double)Depth_select)
                                                                    : std::ceil(merge_node->Dense_input_channel/(double)Depth_select);
    size_t once_cal_weight_num_select = merge_node->kernel_size==3 ? ONCE_CAL_WEIGHT_NUM_3 : ONCE_CAL_WEIGHT_NUM_1;

    // FcNet時不只要考慮ICH_ROUND OCH_ROUND,Input Feature Map 以及 Output Feature Map 數量也需考慮
    const size_t weight_num = (merge_node->class_name!="Dense") ? ((OCH_NUM * ICH_NUM * once_cal_weight_num_select) + (Depth_select*2 * OCH_NUM))
                                                                 : ((OCH_NUM * ICH_NUM * once_cal_weight_num_select * merge_node->Dense_size_input_x * merge_node->Dense_size_input_y * merge_node->Dense_size_output_x * merge_node->Dense_size_output_y)
                                                                 + (Depth_select * 2 * OCH_NUM * merge_node->Dense_size_output_x * merge_node->Dense_size_output_y));
    //const size_t HW_KERNEL_SIZE = 9;
    const size_t HW_KERNEL_SIZE = std::pow(merge_node->kernel_size,2); //這邊有可能是 1 或是 3 現在可以吃進來了
    std::vector<std::ifstream> file_set;
    const size_t need_file_num = merge_node->Have_BatchNormalization ? (merge_node->Have_bias ? 6 : 5 ) : (merge_node->Have_bias ? 2 : 1); //這邊之後要改因為學長這邊搞錯，應該最多有六份才對是weight的bias
    std::vector<short> weight_data(weight_num,0);
    std::cout<<"This Layer weight_data.size : "<<weight_data.size()<<std::endl;
    data_in.reserve(need_file_num);
    file_set.resize(need_file_num);
    if(path_set.size()!=need_file_num){
        throw std::invalid_argument("path_set_num_not equal need_file_num");
    }
    for(size_t i=0;i<need_file_num;i++){
        file_set[i].open(path_set[i]);
        if(!file_set[i].is_open()){
            std::cout<<fmt::format("Can not open {} {} file\n",merge_node->weight_name,weight_type(i,need_file_num))<<std::endl;
            throw std::invalid_argument("Can not open weight file");
        }
    }
    for(size_t i=0;i<need_file_num;i++){
        size_t tmp;
        if(i==0){
            if(merge_node->class_name!="Dense")
                tmp = merge_node->input_channel * merge_node->output_channel * merge_node->kernel_size * merge_node->kernel_size;
            else if(!IF_PRE_NODE_IS_DENSE)
                tmp = merge_node->input_channel * merge_node->Dense_size_input_x * merge_node->Dense_size_input_y * merge_node->output_channel;
            else
                tmp = merge_node->input_channel * merge_node->output_channel;
            //tmp = merge_node->class_name!="Dense" ? merge_node->input_channel * merge_node->output_channel * merge_node->kernel_size * merge_node->kernel_size
            //                                      : merge_node->input_channel * merge_node->Dense_size_input_x * merge_node->Dense_size_input_y * merge_node->output_channel;
            std::cout<<"This is weight number "<<tmp<<std::endl;
        }else{
            tmp = merge_node->output_channel;
        }
        data_in.push_back(get_weight_from_file(file_set[i],tmp));
        file_set[i].close();
    }
    switch(need_file_num){
        case 6:
            File_Attributes.weight   = 0;
            File_Attributes.bias     = 1;
            File_Attributes.beta     = 2;
            File_Attributes.gamma    = 3;
            File_Attributes.variance = 4;
            File_Attributes.mean     = 5;
            break;
        case 5:
            File_Attributes.weight   = 0;
            File_Attributes.beta     = 1;
            File_Attributes.gamma    = 2;
            File_Attributes.variance = 3;
            File_Attributes.mean     = 4;
            break;
        case 2:
            File_Attributes.weight   = 0;
            File_Attributes.bias     = 1;
            break;
        case 1:
            File_Attributes.weight   = 0;
            break;
        default:
            throw std::invalid_argument("sorry you need to check in there，your file number is wrong not 6 or 5 or 2 or 1");
            break;
    }
    size_t times_calc = (merge_node->class_name=="Dense") ? 3 : (merge_node->kernel_size==1 ? 3 : 1); //如果你的kernel_size是3*3的話那你就不用跑三遍，因為你是8 in 8 out，但是如果你是1*1 那你就要一個ch 24   
    std::cout<<"times_calc : "<<times_calc<<std::endl;
    std::cout<<"ICH_NUM    : "<<ICH_NUM<<std::endl;
    std::cout<<"OCH_NUM    : "<<OCH_NUM<<std::endl;
    if(merge_node->class_name=="Dense"){ //choose the algorithm for this merge_node
        algorithm_for_Dense(merge_node,Depth_select,once_cal_weight_num_select,times_calc,OCH_NUM,ICH_NUM,HW_KERNEL_SIZE,data_in,weight_data); 
    }else{
        algorithm_for_basic_conv(merge_node,Depth_select,once_cal_weight_num_select,times_calc,OCH_NUM,ICH_NUM,HW_KERNEL_SIZE,data_in,weight_data);
    }
    std::cout<<weight_data[0]<<std::endl;
    return weight_data;
}
std::vector<std::vector<short>>pstruct::gen_total_weight(const std::string &dir_path,const std::vector<std::shared_ptr<pstruct>> &merge_node_vector){
    std::vector<short> weight_data;
    std::vector<std::vector<short>>weight_data_vector;
    size_t total_weight_count = 0;
    std::map<std::string,bool> weight_table;
    for(auto &merge_node : merge_node_vector){
        const size_t Depth_select = merge_node->class_name=="Dense" ?  DATA_Depth_24 : merge_node->kernel_size==1 ? DATA_Depth_24 : DATA_DEPTH;
        if(!weight_table[merge_node->weight_name]){
            auto once_cal_weight_num_select = (merge_node->kernel_size==3) ? ONCE_CAL_WEIGHT_NUM_3 : ONCE_CAL_WEIGHT_NUM_1;
            total_weight_count  = total_weight_count + std::ceil(merge_node->input_channel/(double)(Depth_select))*std::ceil(merge_node->output_channel/(double)(Depth_select))*once_cal_weight_num_select + std::ceil(merge_node->output_channel/(double)(Depth_select))*Depth_select*2;
            weight_table[merge_node->weight_name] = true;
        }
    }
    weight_data.reserve(total_weight_count);
    for(auto &merge_node : merge_node_vector){
        if(!weight_table[merge_node->weight_name]){
            std::cout<<fmt::format("Processing {} jump jump jump jump jump jump jump jump jump jump",merge_node->weight_name)<<std::endl;
            continue;
        }
        weight_table[merge_node->weight_name] = false;
        std::cout<<fmt::format("Processing {} weight Please Wait!!!!!",merge_node->weight_name)<<std::endl;
        const std::vector<std::string> path_set = merge_node->gen_weight_path(dir_path,merge_node);
        for(auto pola : path_set){
            std::cout<<pola<<std::endl;
        }
        std::cout<<"1.CHECK FILE PATH SUCCESSFUL"<<std::endl;
        auto weight_data_tmp = merge_node->gen_layer_weight(merge_node,path_set);
        std::cout<<weight_data_tmp[0]<<std::endl;
        std::cout<<"2.CHECK WEIGHT_DATA_TMP SUCCESSFIL"<<std::endl;
        //weight_data.insert(weight_data.end(),tmp.begin(),tmp.end()); i close for spilt
        weight_data_vector.push_back(weight_data_tmp);
        std::cout<<"3.CHECK THIS　LAYER WEIGHT GENERATE SUCCESSFUL"<<std::endl;
        std::cout<<std::endl;
    }
    return weight_data_vector;//weight_data
}
std::vector<std::string>pstruct::gen_weight_path(const std::string &dir_path,const std::shared_ptr<pstruct> &merge_node){
    const size_t path_num = merge_node->Have_BatchNormalization ? (merge_node->Have_bias ? 6 : 5) : (merge_node->Have_bias ? 2 : 1);
    std::vector<std::string> path_set;
    std::string tmp;
    path_set.reserve(path_num);
    if(path_num==6){ //代表再訓練那邊全開，六份檔案都要有
        for(size_t i=0;i<path_num;i++){
            switch(i){
                case 0 : //Weight
                    tmp = merge_node->weight_name + "_weight.txt";
                    break;
                case 1 : //Bias
                    tmp = merge_node->weight_name + "_bias.txt";
                    break;
                case 2 : //Beta
                    tmp = merge_node->weight_name + "_beta.txt";
                    break;
                case 3 : //Scale
                    tmp = merge_node->weight_name + "_gamma.txt";
                    break;
                case 4 : //Variance.txt
                    tmp = merge_node->weight_name + "_variance.txt";
                    break;
                case 5 : //Mean
                    tmp = merge_node->weight_name + "_mean.txt";
                    break;
            }
            path_set.push_back(dir_path+"/"+tmp);
        }
    }
    else if(path_num==5){
        for(size_t i=0;i<path_num;i++){
            switch(i){
                case 0 : //Weight
                    tmp = merge_node->weight_name + "_weight.txt";
                    break;
                case 1 : //Beta
                    tmp = merge_node->weight_name + "_beta.txt";
                    break;
                case 2 : //Scale
                    tmp = merge_node->weight_name + "_gamma.txt";
                    break;
                case 3 : //Variance.txt
                    tmp = merge_node->weight_name + "_variance.txt";
                    break;
                case 4 : //Mean
                    tmp = merge_node->weight_name + "_mean.txt";
                    break;
            }
            path_set.push_back(dir_path+"/"+tmp);
        }
    }
    else if(path_num==2){
        for(size_t i=0;i<path_num;i++){
            switch(i){
                case 0 : //Weight
                    tmp = merge_node->weight_name + "_weight.txt";
                    break;
                case 1 : //Bias
                    tmp = merge_node->weight_name + "_bias.txt";
                    break;
            }
            path_set.push_back(dir_path+"/"+tmp);
        }
    }
    else if(path_num==1){
        for(size_t i=0;i<path_num;i++){
            switch(i){
                case 0 : //Weight
                    tmp = merge_node->weight_name + "_weight.txt";
                    break;
            }
            path_set.push_back(dir_path+"/"+tmp);
        }
    }
    else{
        throw std::invalid_argument("sorry you need to check this file about you need not 6 || 5 || 2 || 1 file");
    }
    return path_set;
}
void pstruct::dump_weight_sim(const std::string &filename,const std::vector<short>&weight_data){
    std::ofstream out(filename);
    if(!out.is_open()){
        std::cerr << "Can not open "<< filename <<std::endl;
        throw std::invalid_argument("Open file failed");
    }
    for(const auto &i : weight_data){
        out << fmt::format("{:04x}\n",(ushort)i);
    }
    out.close();
}
void pstruct::dump_weight_bin(const std::string &filename, const void *src, const size_t src_size){
    int fd = open(filename.c_str(), O_CREAT | O_RDWR | O_SYNC, S_IRUSR | S_IWUSR);
    if(fd < 0){
        throw std::invalid_argument("dump_bin : Can not open dump bin file.");
    }
    int write_num = write(fd, src, src_size);
    close(fd);
    if(write_num < 0){
        throw std::invalid_argument("dump_bin : Can not write dump bin file.");
    }
}
template <typename F> std::ostream & operator<<(std::ostream & os,const std::vector<std::vector<F>> &vec){
    int counter = 0;
    for(auto elem : vec){
        os<<counter<<".{ ";
        for(auto elems : elem){
            os<<elems<<" ";
        }
        os<<"}"<<"\n";
        counter++;
    }
    os<<std::flush;
    return os;
}
template <typename T> std::ostream & operator<<(std::ostream & os,const std::vector<T> &vec){
    int counter = 0;
    for(auto elem : vec){
        os<<"index"<<counter<<":"<<elem<<std::endl;
        counter++;
    }
    os<<std::flush;
    return os;
}
template <typename A> std::ostream & operator<<(std::ostream & os,const std::vector<std::shared_ptr<A>> &vec){
    int counter = 0;
    for(auto elem : vec){
        os<<counter<<" "<<"class_name              "<<elem->class_name<<"\n";
        os<<counter<<" "<<"input_channel           "<<elem->input_channel<<"\n";
        os<<counter<<" "<<"input_feature_size      "<<elem->input_feature_size<<"\n";
        os<<counter<<" "<<"output_channel          "<<elem->output_channel<<"\n";
        os<<counter<<" "<<"output_feature_size     "<<elem->output_feature_size<<"\n";
        os<<counter<<" "<<"kernel_size             "<<elem->kernel_size<<"\n";
        os<<counter<<" "<<"kernel_stride           "<<elem->kernel_stride<<"\n";
        os<<counter<<" "<<"padding                 "<<elem->padding<<"\n";
        os<<counter<<" "<<"activate                "<<elem->activate<<"\n";
        os<<counter<<" "<<"pool_size               "<<elem->pool_size<<"\n";
        os<<counter<<" "<<"pool_stride             "<<elem->pool_stride<<"\n";
        os<<counter<<" "<<"pool_padding            "<<elem->pool_padding<<"\n";
        os<<counter<<" "<<"units                   "<<elem->units<<"\n";
        os<<counter<<" "<<"Have_bias               "<<elem->Have_bias<<"\n";
        os<<counter<<" "<<"weight_address          "<<elem->weight_address<<"\n";
        os<<counter<<" "<<"input_padding_size      "<<elem->input_padding_size<<"\n";
        os<<counter<<" "<<"output_padding_size     "<<elem->output_padding_size<<"\n";
        os<<counter<<" "<<"input_tile_number       "<<elem->input_tile_number<<"\n";
        os<<counter<<" "<<"input_tile_size         "<<elem->input_tile_size<<"\n";
        os<<counter<<" "<<"output_tile_number      "<<elem->output_tile_number<<"\n";
        os<<counter<<" "<<"output_tile_size        "<<elem->output_tile_size<<"\n";
        os<<counter<<" "<<"next_tile_size          "<<elem->next_tile_size<<"\n";
        os<<counter<<" "<<"branch_input_tile_size  "<<elem->branch_input_tile_size   <<"\n";
        os<<counter<<" "<<"input_address           "<<elem->input_address<<"\n";
        os<<counter<<" "<<"output_address          "<<elem->output_address<<"\n";
        os<<counter<<" "<<"pool_address            "<<elem->pool_address<<"\n";
        os<<counter<<" "<<"Leaky_ReLU_alpha        "<<std::fixed<<std::setprecision(18)<<elem->Leaky_ReLU_alpha<<"\n";
        os<<counter<<" "<<"Have_BatchNormalization "<<elem->Have_BatchNormalization<<"\n";
        os<<counter<<" "<<"Have_ReLU               "<<elem->Have_ReLU<<"\n";
        os<<counter<<" "<<"Have_Flatten            "<<elem->Have_Flatten<<"\n";      
        os<<counter<<" "<<"Have_Dense              "<<elem->Have_Dense<<"\n";   
        os<<counter<<" "<<"Have_Maxpooling         "<<elem->Have_Maxpooling<<"\n";    
        os<<counter<<" "<<"Have_Concat             "<<elem->Have_Concat<<"\n";    
        os<<counter<<" "<<"Have_Upsample           "<<elem->Have_Upsample<<"\n";    
        os<<counter<<" "<<"Upsample_size           "<<elem->Upsample_size<<"\n";    
        os<<counter<<" "<<"weight name             "<<elem->weight_name<<"\n"; 
        os<<counter<<" "<<"node name               "<<elem->node_name<<"\n"; 
        for(auto x : elem->Previous_node_OCH){
            os<<counter<<" "<<"Previous_node_OCH       "<<x<<"\n";
        }
        os<<counter<<" "<<"Is_LeakyReLU            "<<elem->Is_LeakyReLU<<"\n";     
        os<<counter<<" "<<"Batch_First             "<<elem->Batch_First<<"\n";     
        os<<counter<<" "<<"quant_batch_bias        "<<elem->quant_batch_bias<<"\n";     
        os<<counter<<" "<<"quant_finish            "<<elem->quant_finish<<"\n";     
        os<<counter<<" "<<"pooling_quant_finish    "<<elem->pooling_quant_finish<<"\n";     
        os<<counter<<" "<<"quant_batch             "<<elem->quant_batch<<"\n";     
        os<<counter<<" "<<"quant_word_size         "<<elem->quant_word_size<<"\n";     
        os<<counter<<" "<<"quant_obuf              "<<elem->quant_obuf<<"\n";     
        os<<counter<<" "<<"Concat_output_control   "<<elem->Concat_output_control<<"\n";     
        os<<counter<<" "<<"branch_node             "<<elem->branch_node<<"\n";     
        os<<counter<<" "<<"Dense_size_input_x      "<<elem->Dense_size_input_x<<"\n";     
        os<<counter<<" "<<"Dense_size_input_y      "<<elem->Dense_size_input_y<<"\n";     
        os<<counter<<" "<<"Dense_size_output_x     "<<elem->Dense_size_output_x<<"\n";     
        os<<counter<<" "<<"Dense_size_output_y     "<<elem->Dense_size_output_y<<"\n";     
        os<<counter<<" "<<"Dense_input_node        "<<elem->Dense_input_node<<"\n";     
        os<<counter<<" "<<"Dense_output_node       "<<elem->Dense_output_node<<"\n";     
        os<<counter<<" "<<"Dense_input_channel     "<<elem->Dense_input_channel<<"\n";     
        os<<counter<<" "<<"Dense_output_channel    "<<elem->Dense_output_channel<<"\n";     
        os<<counter<<" "<<"IF_PRE_NODE_IS_DENSE    "<<elem->IF_PRE_NODE_IS_DENSE<<"\n";     
        os<<"-----------------------------------------------------"<<"\n";
        counter++;
    }
    os<<std::flush;
    return os;
}
int main(int argc, char **argv){
    std::cout<<"--------------------------------------------------------------------"<<std::endl;
    std::cout<<"----------------------model_yolo256_start_parser--------------------"<<std::endl;
    std::cout<<"--------------------------------------------------------------------"<<std::endl;
    pstruct  pola;
    //auto [layer_info,layer_target,layer_quant,output_layers]= pola.read_json("./model_yolo256_new.json","./model_yolo256_design.csv");  //read json file
    auto [layer_info,layer_target,layer_quant,output_layers]= pola.read_json(argv[1],argv[2]);  //read json file
    pola.layer_info   = layer_info;
    pola.layer_target = layer_target;
    pola.layer_quant  = layer_quant;
    pola.output_layers = output_layers;
    for(auto output_node_vec : pola.output_layers){//這邊會幫忙抓出那些節點走過那代表他做分支，concat_output_control要拉起來
        for(auto output_node : output_node_vec){
            pola.scan_result_map[output_node] = 0; //最後輸出的點一定只會走一次
            pola.scan_table(pola.layer_info,output_node);
            break;
        }
    }
    //for(auto node : pola.concat_map_vector){
        //std::cout<<node.first<<" "<<node.second<<std::endl;
    //}
    std::cout<<pola.layer_target<<std::endl;
    std::cout<<pola.layer_quant<<std::endl;
    std::cout<<pola.output_layers<<std::endl;
    /*step1*/pola.layer_info_data_vector = pola.split_layer_info_to_vector(pola.layer_info);
    /*step2*/pola.layer_info_data_pointer_vector = pola.layer_info_data_vector_trace(pola.layer_info_data_vector);
    /*step3*/pola.target_layer_info_vector = target_node_trace(pola.layer_target);
    /*step4*/pola.target_layer_quant_vector = target_quant_trace(pola.layer_quant);
    /*step5*/pola.weight_offset(pola.layer_info_data_pointer_vector,pola.target_layer_info_vector);//void just to do information sort and recursive
    /*step6*/pola.merge_node_vector = pola.merge_node(pola.layer_info_data_pointer_vector,pola.target_layer_info_vector,pola.target_layer_quant_vector,pola.merge_node_jump_location);
    /*step7*/pola.merge_node_fix(pola.merge_node_vector,pola.merge_node_jump_location,pola.branch_node_jump_location);
    std::ofstream ofs;
    ofs.open("merge_node_info.txt");
    ofs<<pola.merge_node_vector<<"\n"; //u can use this to cout the every node information
    ofs.close();
    /*step8*/pola.get_tile_info(pola.merge_node_vector);
    /*step9*/
    //DUMP_TILE INFO
    
    #ifdef DUMP_TILE_INFO
        #ifdef DUMP_FOR_SIM
            std::cout<<"dump tile info now!!!!!!"<<std::endl;
            //pola.dump_total_tile_sim("pola_total_tile_info.txt",pola.merge_node_vector);
            for(int i=0;i<pola.merge_node_vector.size();i++){
                std::vector<std::shared_ptr<pstruct>> merge_node;
                merge_node.push_back(pola.merge_node_vector[i]);
                pola.dump_total_tile_sim("./layer_by_layer_tile_info/layer_"+std::to_string(i+1)+"_tile_info.txt",merge_node);
            }
        #else
            std::cout<<"dump tile info BIN now!!!!!!"<<std::endl;
            pola.dump_tile_bin("pola_total_tile_info.bin",pola.merge_node_vector);
        #endif
    #endif    
    //DUMP_LAYER INFO
    #ifdef DUMP_LAYER_INFO
        auto layer_info_data = pola.gen_layer_info_data(pola.merge_node_vector);
        #ifdef DUMP_FOR_SIM
            std::cout<<"dump layer info now!!!!!!"<<std::endl;
            pola.dump_layer_info_sim("pola_layer_info.txt",layer_info_data);
        #else
            std::cout<<"dump layer info BIN now!!!!!!"<<std::endl;
            pola.dump_layer_info_data_bin("pola_layer_info.bin", layer_info_data);
        #endif
    #endif
    #ifdef DUMP_WEIGHT
        std::vector<std::vector<short>> total_weight = pola.gen_total_weight(argv[3],pola.merge_node_vector);
        #ifdef DUMP_FOR_SIM
            std::cout<<"dump weight info now!!!!!!"<<std::endl;
            for(auto i =0;i<total_weight.size();i++){
                std::vector<short>total_weight_split = total_weight[i];
                std::cout<<total_weight[i][0]<<std::endl;
                pola.dump_weight_sim("./layer_by_layer_weight_info/layer_"+std::to_string(i+1)+"_weight.txt",total_weight_split);
            }
        #else
            std::cout<<"dump weight info BIN now!!!!!!"<<std::endl;
            size_t total_weight_size = 0;            
            std::vector<short> total_weight_pointer;
            for(auto total_weight_index : total_weight){
                total_weight_size += size_in_byte(total_weight_index);
		        total_weight_pointer.insert(total_weight_pointer.end(),total_weight_index.begin(),total_weight_index.end());
            }
            pola.dump_weight_bin("pola_total_weight.bin", total_weight_pointer.data(), total_weight_size);
        #endif
    #endif

    std::ofstream out("pola_Output_Offset.txt");
    //for(std::shared_ptr<pstruct> &merge_node : pola.merge_node_vector){
    for(auto i=0;i<pola.merge_node_vector.size();i++){
        #ifdef DUMP_FOR_SIM
            //out<<fmt::format("{:08x}\n",merge_node->tile_info[0]+DATA_BASE_ADDR);
            out<<fmt::format("{:08x}\n",pola.merge_node_vector[i]->tile_info[0]+DATA_BASE_ADDR);
        #else
            if(output_layers.empty() && i==pola.merge_node_vector.size()-1){
                out<<fmt::format("{}\n",pola.merge_node_vector[i]->tile_info[0]);
            }else{
                for(auto j:pola.output_layers){
                    if(pola.merge_node_vector[i]->weight_name==j[0]){
			std::cout<<j<<std::endl;
                        out<<fmt::format("{}\n",pola.merge_node_vector[i]->tile_info[0]);
                    }
                }
            }
        #endif
    }
    std::cout<<"----------------------FINIFSH--------------------"<<std::endl;
    std::cout<<"----------------------FINIFSH--------------------"<<std::endl;
    std::cout<<"----------------------FINIFSH--------------------"<<std::endl;
    out.close();
}
