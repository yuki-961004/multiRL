#include <Rcpp.h>
#include "../inst/include/tool_record_shown.hpp" 
#include "../inst/include/tool_sample_choice.hpp" 

//' multiRL.output
//'
//' @param record multiRL.record
//' @param extra A list of extra information passed from R.
//' @return An S4 object of class \code{multiRL.output}.
//' 
//'  \describe{
//'     \item{\code{input}}{
//'       An object of class \code{multiRL.input},
//'       containing the raw data, column specifications, parameters and ...
//'     }
//'     \item{\code{behrule}}{
//'       An object of class \code{multiRL.behrule},
//'       defining the latent learning rules.
//'     }
//'     \item{\code{result}}{
//'       An object of class \code{multiRL.result},
//'       storing trial-level outputs of the Markov Decision Process.
//'     }
//'     \item{\code{extra}}{
//'       A \code{List} containing additional user-defined information.
//'     }
//'   }
//' 
//' @export
// [[Rcpp::export]]
Rcpp::S4 process_4_output_cpp(const Rcpp::S4 record, const Rcpp::List& extra) {

/******************************* [load input] *********************************/

    const Rcpp::S4 input = record.slot("input");

    // R: record@input@settings
    const Rcpp::S4 settings = input.slot("settings");
    const std::string policy = settings.slot("policy");
    const Rcpp::CharacterVector system = settings.slot("system");

    // R: record@input@features@idinfo
    const Rcpp::S4 features = input.slot("features");
    const Rcpp::CharacterMatrix idinfo = Rcpp::as<Rcpp::CharacterMatrix>(
        features.slot("idinfo")
    );
    const Rcpp::CharacterVector subid  = idinfo(Rcpp::_, 0);
    const Rcpp::CharacterVector block  = idinfo(Rcpp::_, 1);
    const Rcpp::CharacterVector trial  = idinfo(Rcpp::_, 2);

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
    const Rcpp::S4 params_raw = input.slot("params");
    Rcpp::Function c_r("c");
    const Rcpp::List params = c_r(
        params_raw.slot("free"), 
        params_raw.slot("fixed"), 
        params_raw.slot("constant")
    );

    // R: record@input@funcs
    const Rcpp::S4 funcs = input.slot("funcs");
    const Rcpp::Function rate_func(funcs.slot("rate_func"));
    const Rcpp::Function prob_func(funcs.slot("prob_func"));
    const Rcpp::Function util_func(funcs.slot("util_func"));
    const Rcpp::Function bias_func(funcs.slot("bias_func"));
    const Rcpp::Function expl_func(funcs.slot("expl_func"));
    const Rcpp::Function dcay_func(funcs.slot("dcay_func"));

/****************************** [load behrule] ********************************/
    
    const Rcpp::S4 behrule = record.slot("behrule");

    // R: record@behrule
    const Rcpp::CharacterVector cue = Rcpp::as<Rcpp::CharacterVector>(
        behrule.slot("cue")
    );
    const Rcpp::CharacterVector rsp = Rcpp::as<Rcpp::CharacterVector>(
        behrule.slot("rsp")
    );

    // number of ...    
    int n_rows = Rcpp::as<int>( input.slot("n_rows") );
    int n_cues = cue.size(); 
    int n_rsps = rsp.size();
    int n_system = system.size();

/******************************* [load record] ********************************/
  
    Rcpp::S4 result = Rcpp::clone(
        Rcpp::as<Rcpp::S4>(record.slot("result")) 
    );

    // R: record@result [action select]
    Rcpp::List value = Rcpp::clone(Rcpp::as<Rcpp::List>(
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
    Rcpp::CharacterMatrix position = Rcpp::clone(Rcpp::as<Rcpp::CharacterMatrix>(
        result.slot("position")
    )); 

    // behave
    Rcpp::CharacterMatrix behave_raw = Rcpp::cbind(
        Rcpp::as<Rcpp::CharacterMatrix>(features.slot("action")),
        Rcpp::as<Rcpp::CharacterMatrix>(result.slot("latent")),
        Rcpp::as<Rcpp::CharacterMatrix>(result.slot("simulation")),
        Rcpp::as<Rcpp::CharacterMatrix>(result.slot("position"))
    );

    // 给behave赋予第0行
    Rcpp::CharacterMatrix behave(n_rows + 1, 4);
    behave.row(0) = Rcpp::CharacterVector::create(
        NA_STRING, NA_STRING, NA_STRING, NA_STRING
    );
    // 将原矩阵数据拷贝到新矩阵
    for (int i = 0; i < n_rows; i++) {
        behave.row(i + 1) = behave_raw.row(i);
    }

    Rcpp::colnames(behave) = Rcpp::CharacterVector::create(
        "action", "latent", "simulation", "position"
    );

/******************************* [load others] ********************************/

    // cue建立哈希表
    std::unordered_map<std::string, int> cue_map;
    cue_map.reserve(n_cues);
    for (int j = 0; j < n_cues; j++) {
        cue_map[ std::string(CHAR(cue[j])) ] = j;
    }

    // rsp建立哈希表
    std::unordered_set<std::string> rsp_set;
    rsp_set.reserve(n_rsps);

    for (int i = 0; i < n_rsps; i++) {
        rsp_set.insert( Rcpp::as<std::string>(rsp[i]) );
    }

/****************************** [initial value] *******************************/

    int seed = params["seed"];
    double Q0 = params["Q0"];
    double reset = params["reset"];

    Rcpp::Function r_rbind("rbind");
    Rcpp::NumericVector new_row(n_cues, NA_REAL);

    for (int i = 0; i < value.size(); i++) {
        Rcpp::NumericMatrix sub_value = value[i];
        std::fill( 
            sub_value.row(0).begin(), sub_value.row(0).end(), 
            std::isnan(Q0) ? 0.0 : Q0 
        );
        sub_value = Rcpp::as<Rcpp::NumericMatrix>( r_rbind(sub_value, new_row) );
        value[i] = sub_value;
    }
    
    std::fill( count.row(0).begin(), count.row(0).end(), 0.0 );
    count = Rcpp::as<Rcpp::NumericMatrix>( r_rbind(count, new_row) );

/******************************** [main loop] *********************************/

    // 设置随机种子
    Rcpp::Function r_set_seed("set.seed");
    r_set_seed(seed);

    for (int i = 0; i < n_rows; i++) {

/******************************* [action select] ******************************/

        // 记录刺激是否在该试次(state)出现
        shown.row(i) = record_shown(state[i], cue);

        // bias function: 每个刺激上的偏见
        bias.row(i) = Rcpp::as<Rcpp::NumericVector>(
            bias_func(
                Rcpp::_["shown"]  = Rcpp::NumericVector(shown.row(i)),
                Rcpp::_["count"]  = Rcpp::NumericVector(count.row(i)),
                Rcpp::_["params"] = params,
                Rcpp::_["idinfo"] = Rcpp::CharacterVector(idinfo.row(i)),
                Rcpp::_["exinfo"] = Rcpp::CharacterVector(exinfo.row(i)),
                Rcpp::_["behave"] = Rcpp::CharacterVector(behave.row(i)),
                Rcpp::_["cue"] = cue, Rcpp::_["rsp"] = rsp, 
                Rcpp::_["state"] = state[i]
            )
        );
        // exploration function: 此次是否进行探索
        exploration.row(i) = Rcpp::as<Rcpp::NumericVector>(
            expl_func(
                Rcpp::_["shown"]  = Rcpp::NumericVector(shown.row(i)),
                Rcpp::_["rownum"] = i + 1,
                Rcpp::_["params"] = params,
                Rcpp::_["idinfo"] = Rcpp::CharacterVector(idinfo.row(i)),
                Rcpp::_["exinfo"] = Rcpp::CharacterVector(exinfo.row(i)),
                Rcpp::_["behave"] = Rcpp::CharacterVector(behave.row(i)),
                Rcpp::_["cue"] = cue, Rcpp::_["rsp"] = rsp, 
                Rcpp::_["state"] = state[i]
            )
        );   
        // probability function: 选择每个选项的概率 
        Rcpp::List qvalue(n_system);

        for (int s = 0; s < n_system; s++) {
            Rcpp::NumericMatrix sub_value = value[s];
            Rcpp::NumericVector sub_qvalue(n_cues);
            for (int j = 0; j < n_cues; j++) {
                sub_qvalue[j] =
                    ( sub_value(i, j) + bias(i, j) ) * shown(i, j);
            }
            qvalue[s] = sub_qvalue;
        }

        prob.row(i) = Rcpp::as<Rcpp::NumericVector>(
            prob_func(
                Rcpp::_["shown"]  = Rcpp::NumericVector(shown.row(i)),
                Rcpp::_["qvalue"] = qvalue,
                Rcpp::_["explor"] = Rcpp::NumericVector(exploration.row(i)),
                Rcpp::_["params"] = params,
                Rcpp::_["system"] = system,
                Rcpp::_["idinfo"] = Rcpp::CharacterVector(idinfo.row(i)),
                Rcpp::_["exinfo"] = Rcpp::CharacterVector(exinfo.row(i)),
                Rcpp::_["behave"] = Rcpp::CharacterVector(behave.row(i)),
                Rcpp::_["cue"] = cue, Rcpp::_["rsp"] = rsp, 
                Rcpp::_["state"] = state[i]
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

        position(i, 0) = std::to_string(row_index + 1);
        // 记录当前行为到当前试次, 会覆盖上一次的行为
        behave(i, 1) = latent(i, 0);
        behave(i, 2) = simulation(i, 0);
        behave(i, 3) = position(i, 0);
        // 记录当前行为到下一个试次, 用于action select三函数读取
        behave(i + 1, 1) = latent(i, 0);
        behave(i + 1, 2) = simulation(i, 0);
        behave(i + 1, 3) = position(i, 0);

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
                Rcpp::_["shown"]  = Rcpp::NumericVector(shown.row(i)),
                Rcpp::_["reward"] = Rcpp::NumericVector(reward.row(i)),
                Rcpp::_["params"] = params,
                Rcpp::_["idinfo"] = Rcpp::CharacterVector(idinfo.row(i)),
                Rcpp::_["exinfo"] = Rcpp::CharacterVector(exinfo.row(i)),
                Rcpp::_["behave"] = Rcpp::CharacterVector(behave.row(i)),
                Rcpp::_["cue"] = cue, Rcpp::_["rsp"] = rsp, 
                Rcpp::_["state"] = state[i]
            )
        ); 

        // 提取此次选择的latent为target
        const std::string target = Rcpp::as<std::string>( latent(i,0) );
        // 在cue中寻找target
        col_index = cue_map[target];
        
        bool is_nb;
        if (!std::isnan(reset)) {
            is_nb = (i > 0) && (block[i] != block[i - 1]);
        } else {
            is_nb = false;  
        }

        // 检查此时是否是第一次选(全局第一次 or 局部第一次, 都算)
        bool is_fp = (count(i, col_index) == 0);

        for (int s = 0; s < n_system; s++) {

            std::string sub_system = Rcpp::as<std::string>(system[s]);

            Rcpp::NumericMatrix sub_value = value[sub_system];

            double Qi;
            Rcpp::NumericVector cur_value;

            // 是否在进入新 block 时重置
            if (is_nb) {
                cur_value = Rcpp::rep(reset, sub_value.ncol());
                Qi        = reset;
            } else {
                cur_value = Rcpp::NumericVector(sub_value.row(i));
                Qi        = sub_value(i, col_index);
            }

            // decay：未被选择选项的价值衰减
            sub_value.row(i + 1) =
                Rcpp::as<Rcpp::NumericVector>(
                    dcay_func(
                        Rcpp::_["shown"]  = Rcpp::NumericVector(shown.row(i)),
                        Rcpp::_["value0"] = Rcpp::NumericVector(sub_value.row(0)),
                        Rcpp::_["values"] = cur_value,
                        Rcpp::_["reward"] = Rcpp::NumericVector(reward.row(i)),
                        Rcpp::_["utility"] = utility(i, 0),
                        Rcpp::_["params"] = params,
                        Rcpp::_["system"] = sub_system,
                        Rcpp::_["idinfo"] = Rcpp::CharacterVector(idinfo.row(i)),
                        Rcpp::_["exinfo"] = Rcpp::CharacterVector(exinfo.row(i)),
                        Rcpp::_["behave"] = Rcpp::CharacterVector(behave.row(i)),
                        Rcpp::_["cue"] = cue, Rcpp::_["rsp"] = rsp, 
                        Rcpp::_["state"] = state[i]
                    )
                );

            // learning rate 更新
            if (std::isnan(Q0) && is_fp) {
                sub_value(i + 1, col_index) = utility(i, 0);
                sub_value(0,     col_index) = utility(i, 0);
            } else {
                sub_value(i + 1, col_index) =
                    Rcpp::as<double>(
                        rate_func(
                            Rcpp::_["shown"]  = Rcpp::NumericVector(shown.row(i)),
                            Rcpp::_["qvalue"] = Qi,
                            Rcpp::_["reward"] = Rcpp::NumericVector(reward.row(i)),
                            Rcpp::_["utility"] = utility(i, 0),
                            Rcpp::_["params"] = params,
                            Rcpp::_["system"] = sub_system,
                            Rcpp::_["idinfo"] = Rcpp::CharacterVector(idinfo.row(i)),
                            Rcpp::_["exinfo"] = Rcpp::CharacterVector(exinfo.row(i)),
                            Rcpp::_["behave"] = Rcpp::CharacterVector(behave.row(i)),
                            Rcpp::_["cue"] = cue, Rcpp::_["rsp"] = rsp, 
                            Rcpp::_["state"] = state[i]
                        )
                    );
            }

            // 写回 list
            value[sub_system] = sub_value;
        }
        
        //如果需要重置, 且进入了新block, 则计数器也要归零
        if (is_nb) {
            std::fill( count.row(i+1).begin(), count.row(i+1).end(), 0.0 );
        } else {
            count.row(i+1) = count.row(i);
        }
        
        count(i+1, col_index) = count(i+1, col_index)+1;
    }

/******************************** [delete row 1] ******************************/

    Rcpp::Range rows_to_keep(1, n_rows);
    
    for (int s = 0; s < n_system; s++) {
        Rcpp::NumericMatrix sub_value = value[s];
        sub_value = sub_value(rows_to_keep, Rcpp::_);
        Rcpp::colnames(sub_value) = cue;
        value[s] = sub_value;
    }

    count = count(rows_to_keep, Rcpp::_);
    Rcpp::colnames(count) = cue;

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
    result.slot("position")     = position;

/********************************* [save output] ******************************/

    Rcpp::S4 output("multiRL.output");  

    output.slot("input")        = record.slot("input");
    output.slot("behrule")      = record.slot("behrule");
    output.slot("result")       = result;
    output.slot("extra")        = record.slot("extra");

    return output;
}