#include <Rcpp.h>
#include "tool_get_param.h" 
#include "tool_record_shown.h" 
#include "tool_sample_choice.h" 

//' multiRL.output
//'
//' This function processes the multiRL record and generates output.
//' It accepts an S4 object `record` and a list `extra` for additional parameters.
//'
//' @param record An S4 object containing the multiRL record.
//' @param extra A list of extra parameters passed from R.
//' @return multiRL.output
//' @export
// [[Rcpp::export]]
Rcpp::S4 process_4_output_cpp(const Rcpp::S4 record, const Rcpp::List& extra) {

/******************************* [load input] *********************************/

    const Rcpp::S4 input = record.slot("input");

    // R: record@input@settings
    const Rcpp::S4 settings = input.slot("settings");
    const std::string policy = settings.slot("policy");

    // R: record@input@features@idinfo
    const Rcpp::S4 features = input.slot("features");
    const Rcpp::CharacterMatrix idinfo = Rcpp::as<Rcpp::CharacterMatrix>(
        features.slot("idinfo")
    );

    // R: record@input@features@state
    const Rcpp::CharacterVector state_raw = features.slot("state");
    const Rcpp::IntegerVector dims = state_raw.attr("dim");

    int dim1 = dims[0];
    int dim2 = dims[1];
    int dim3 = dims[2];

    Rcpp::List state(dim1);

    for (int i = 0; i < dim1; i++) {
        Rcpp::CharacterMatrix mat(dim2, dim3);
        for (int j = 0; j < dim2; j++) {
            for (int k = 0; k < dim3; k++) {
                int idx = i + j * dim1 + k * dim1 * dim2;  
                mat(j, k) = state_raw[idx];
            }
        }
        state[i] = mat;
    }

    // R: record@input@features@action
    const Rcpp::CharacterMatrix action = Rcpp::as<Rcpp::CharacterMatrix>(
        features.slot("action")
    );
    // R: record@input@features@exinfo
    const Rcpp::CharacterMatrix exinfo = Rcpp::as<Rcpp::CharacterMatrix>(
        features.slot("exinfo")
    );
  
    // R: record@input@params
    const Rcpp::S4 params = input.slot("params");

    // R: record@input@funcs
    const Rcpp::S4 funcs = input.slot("funcs");
    const Rcpp::Function rate_func(funcs.slot("rate_func"));
    const Rcpp::Function prob_func(funcs.slot("prob_func"));
    const Rcpp::Function util_func(funcs.slot("util_func"));
    const Rcpp::Function bias_func(funcs.slot("bias_func"));
    const Rcpp::Function expl_func(funcs.slot("expl_func"));

/****************************** [load behrule] ********************************/
    
    const Rcpp::S4 behrule = record.slot("behrule");

    // R: record@behrule
    const Rcpp::CharacterVector cue = Rcpp::as<Rcpp::CharacterVector>(
        behrule.slot("cue")
    );
    const Rcpp::CharacterVector rsp = Rcpp::as<Rcpp::CharacterVector>(
        behrule.slot("rsp")
    );

/******************************* [load record] ********************************/
  
    Rcpp::S4 result = Rcpp::clone(
        Rcpp::as<Rcpp::S4>(record.slot("result")) 
    );

    // R: record@result [action select]
    Rcpp::NumericMatrix value = Rcpp::clone(Rcpp::as<Rcpp::NumericMatrix>(
        result.slot("value")
    ));
    Rcpp::NumericMatrix bias = Rcpp::clone(Rcpp::as<Rcpp::NumericMatrix>(
        result.slot("bias")
    ));
    Rcpp::NumericMatrix shown = Rcpp::clone(Rcpp::as<Rcpp::NumericMatrix>(
        result.slot("shown")
    ));
    Rcpp::NumericMatrix prob = Rcpp::clone(Rcpp::as<Rcpp::NumericMatrix>(
        result.slot("prob")
    ));
    Rcpp::NumericMatrix count = Rcpp::clone(Rcpp::as<Rcpp::NumericMatrix>(
        result.slot("count")
    ));

    // R: record@result [value update]
    Rcpp::NumericMatrix exploration = Rcpp::clone(Rcpp::as<Rcpp::NumericMatrix>(
        result.slot("exploration")
    ));
    Rcpp::CharacterMatrix latent = Rcpp::clone(Rcpp::as<Rcpp::CharacterMatrix>(
        result.slot("latent")
    ));  
    Rcpp::NumericMatrix reward = Rcpp::clone(Rcpp::as<Rcpp::NumericMatrix>(
        result.slot("reward")
    ));  
    Rcpp::NumericMatrix utility = Rcpp::clone(Rcpp::as<Rcpp::NumericMatrix>(
        result.slot("utility")
    ));
    Rcpp::CharacterMatrix simulation = Rcpp::clone(Rcpp::as<Rcpp::CharacterMatrix>(
        result.slot("simulation")
    ));  

/******************************* [load others] ********************************/

    int n_rows = Rcpp::as<int>( input.slot("n_rows") );
    int n_cues = cue.size(); 
    int n_rsps = rsp.size();

    // 建立哈希表：cue 名称 → index
    std::unordered_map<std::string, int> cue_map;
    cue_map.reserve(n_cues);
    for (int j = 0; j < n_cues; j++) {
        cue_map[ std::string(CHAR(cue[j])) ] = j;
    }

    // 把 rsp 向量预处理成 unordered_set，加速查找
    std::unordered_set<std::string> rsp_set;
    rsp_set.reserve(n_rsps);

    for (int i = 0; i < n_rsps; i++) {
        rsp_set.insert( Rcpp::as<std::string>(rsp[i]) );
    }

/****************************** [initial value] *******************************/

    double Q1 = get_param(params, "Q1");
    if (std::isnan(Q1)) {Q1 = 0.0;}
    std::fill( value.row(0).begin(), value.row(0).end(), Q1 );
    std::fill( count.row(0).begin(), count.row(0).end(), 0.0 );

    Rcpp::Function r_rbind("rbind");

    Rcpp::NumericVector new_row(n_cues, NA_REAL);
    value = Rcpp::as<Rcpp::NumericMatrix>( r_rbind(value, new_row) );
    count = Rcpp::as<Rcpp::NumericMatrix>( r_rbind(count, new_row) );

/******************************** [main loop] *********************************/

    // 设置随机种子
    Rcpp::Function r_set_seed("set.seed");
    r_set_seed(123);

    for (int i = 0; i < n_rows; i++) {

/******************************* [action select] ******************************/

        // 记录每个刺激是否出现
        shown.row(i) = record_shown(state[i], cue);

        // bias function: 每个刺激上的偏见
        bias.row(i) = Rcpp::as<Rcpp::NumericVector>(
            bias_func(
                Rcpp::_["count"]  = Rcpp::NumericVector(count.row(i)),
                Rcpp::_["params"] = params,
                Rcpp::_["idinfo"] = Rcpp::CharacterVector(idinfo.row(i)),
                Rcpp::_["exinfo"] = Rcpp::CharacterVector(exinfo.row(i))
            )
        );
        // exploration function: 此次是否进行探索
        exploration.row(i) = Rcpp::as<Rcpp::NumericVector>(
            expl_func(
                Rcpp::_["rownum"] = i,
                Rcpp::_["params"] = params,
                Rcpp::_["idinfo"] = Rcpp::CharacterVector(idinfo.row(i)),
                Rcpp::_["exinfo"] = Rcpp::CharacterVector(exinfo.row(i))
            )
        );   
        // probability function: 选择每个选项的概率 
        Rcpp::NumericVector qvalue(n_cues);

        for (int j = 0; j < n_cues; j++) {
            qvalue[j] = ( value(i, j) + bias(i, j) ) * shown(i, j);
        }   
        
        prob.row(i) = Rcpp::as<Rcpp::NumericVector>(
            prob_func(
                Rcpp::_["qvalue"] = qvalue,
                Rcpp::_["explor"] = Rcpp::NumericVector(exploration.row(i)),
                Rcpp::_["params"] = params,
                Rcpp::_["idinfo"] = Rcpp::CharacterVector(idinfo.row(i)),
                Rcpp::_["exinfo"] = Rcpp::CharacterVector(exinfo.row(i))
            )
        );

/************************************ [policy] ********************************/

        // 寻找行为对应的奖励
        int row_index = -1;
        int col_index = -1;
        
        Rcpp::CharacterMatrix state_i = state[i];

        if (policy == "off") {

            // 复制人类的行为
            latent(i,0) = action(i,0);
            // 此次选择的latent记为target
            const std::string target = Rcpp::as<std::string>( latent(i,0) );
            // 在state中找latent
            for (int r = 0; r < state_i.nrow(); r++) {
                for (int c = 0; c < state_i.ncol()-1; c++) {

                    const char* cell_ptr = CHAR( state_i(r, c) );  
                    if (target == cell_ptr) {row_index = r; break;}

                } 
                if (row_index != -1) break;
            }

            simulation(i,0) = action(i,0);

        } else if (policy == "on") {

            // 根据概率产生随机选择
            latent(i, 0) = sample_choice(prob.row(i), shown.row(i), cue);
            // 此次选择的latent记为target
            const std::string target = Rcpp::as<std::string>(latent(i,0));
            // 在state中找latent
            for (int r = 0; r < state_i.nrow(); r++) {
                for (int c = 0; c < state_i.ncol()-1; c++) {
                    // 存储state所有字符串
                    const char* cell_ptr = CHAR( state_i(r, c) );  
                    // 在state中搜索target
                    if (target == cell_ptr) {
                        row_index = r;
                        break;
                    }
                }
                if (row_index != -1) break;
            }
            // 找到latent对应的simulation
            for (int c = 0; c < state_i.ncol()-1; c++) {
                // 存储latent所在行
                const char* cell_ptr = CHAR( state_i(row_index, c) );  
                // 在latent所在行找rsp
                if (rsp_set.find(cell_ptr) != rsp_set.end()) {
                    simulation(i, 0) = cell_ptr;
                    break;
                }
            }
        }

        // 最后一列是奖励数值
        int col_reward = state_i.ncol()-1;   

        // 将state中存放的字符串类型奖励转换成数值
        double reward_val = std::stod(
            Rcpp::as<std::string>( state_i(row_index, col_reward) )
        );
        reward(i, 0) = reward_val;

/******************************** [value update] ******************************/

        // utility function: 奖励转换成主观价值
        utility.row(i) = Rcpp::as<Rcpp::NumericVector>(
            util_func(
                Rcpp::_["reward"] = Rcpp::NumericVector(reward.row(i)),
                Rcpp::_["params"] = params,
                Rcpp::_["idinfo"] = Rcpp::CharacterVector(idinfo.row(i)),
                Rcpp::_["exinfo"] = Rcpp::CharacterVector(exinfo.row(i))
            )
        ); 

        // 记录上次价值
        value.row(i+1) = value.row(i);
        // 提取此次选择的latent为target
        const std::string target = Rcpp::as<std::string>( latent(i,0) );
        // 在cue中寻找target
        col_index = cue_map[target];
        // 记录此时价值
        double Qi = value(i, col_index);

        // learning rate function: 以什么比例采信预测误差
        if (Rcpp::NumericVector::is_na(Q1) && Qi == 0) {
            // learning rate = 100%
            value(i+1, col_index) = utility(i, 0);
        } else {
            // learning rate = alpha
            value(i+1, col_index) = Rcpp::as<double>(
                rate_func(
                    Rcpp::_["qvalue"] = value(i, col_index),
                    Rcpp::_["reward"] = utility(i, 0),
                    Rcpp::_["params"] = params,
                    Rcpp::_["idinfo"] = Rcpp::CharacterVector(idinfo.row(i)),
                    Rcpp::_["exinfo"] = Rcpp::CharacterVector(exinfo.row(i))
                )
            );
        }

        count.row(i+1) = count.row(i);
        count(i+1, col_index) = count(i+1, col_index)+1;
    }

/******************************** [delete row 1] ******************************/

    Rcpp::Range rows_to_keep(1, n_rows);
    
    value = value(rows_to_keep, Rcpp::_);
    count = count(rows_to_keep, Rcpp::_);

/********************************* [save result] ******************************/

    result.slot("value")        = value;
    result.slot("bias")         = bias;
    result.slot("shown")        = shown;
    result.slot("prob")         = prob;
    result.slot("count")        = count;

    result.slot("exploration")  = exploration;
    result.slot("latent")       = latent;
    result.slot("reward")       = reward;
    result.slot("utility")      = utility;
    result.slot("simulation")   = simulation;

/********************************* [save output] ******************************/

    Rcpp::S4 output("multiRL.output");  

    output.slot("input")        = record.slot("input");
    output.slot("behrule")      = record.slot("behrule");
    output.slot("result")       = result;
    output.slot("extra")        = record.slot("extra");

    return output;
}