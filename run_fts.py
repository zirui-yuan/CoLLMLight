import os
import time
import argparse
from utils import error
from framework.FTSample import *
from utils.config import *
from utils.utils import merge

def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--memo", type=str, default='MALLMlight-GPT3.5')
    # parser.add_argument("--model_type", type=str, default="gpt") #/gpt/llama
    # parser.add_argument("--model_path", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--memo", type=str, default='FTs_avgql')
    # parser.add_argument("--memo", type=str, default='default strategy')
    parser.add_argument("--model_type", type=str, default="gpt") #/gpt/llama
    parser.add_argument("--model_path", type=str, default=None) 
    parser.add_argument("--num_rounds", type=int, default=1)
    parser.add_argument("--new_max_tokens", type=int, default=2048)
    parser.add_argument("--proj_name", type=str, default="LLM-TSCS")
    parser.add_argument("--eightphase", action="store_true", default=False)
    parser.add_argument("--multi_process", action="store_true", default=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dataset", type=str, default="synthetic_4x4")
    parser.add_argument("--traffic_file", type=str, default="anon_4_4_synthetic_8000.json")
    parser.add_argument("--mode", type=str, default="fts", help="fts")
    parser.add_argument("--reward_type", type=str, default="avg_ql", help="near_ql/near_wt/near_tt/avg_ql/avg_wt/avg_tt/far_ql/far_wt/far_tt/qlwt/early_ocp/far_ocp/avg_ocp")
    parser.add_argument("--long_info", type=bool, default=True)
    parser.add_argument("--feed_back", type=bool, default=True)
    parser.add_argument("--feed_back_num", type=int, default=1)
    parser.add_argument("--debug_wollm", type=bool, default=False)
    parser.add_argument("--reward_period", type=int, default=5)
    parser.add_argument("--ignore_threshold", type=float, default=0.5)
    return parser.parse_args()


def main(in_args):
    traffic_file_list = [] 
    print(in_args.model_path)

    if in_args.dataset == 'jinan':
        count = 3600
        # count = 86400
        road_net = "3_4"
        traffic_file_list = ["anon_3_4_jinan_real.json", "anon_3_4_jinan_real_2000.json",
                             "anon_3_4_jinan_real_2500.json", "anon_3_4_jinan_synthetic_24000_60min.json", 
                             "anon_3_4_jinan_synthetic_24h.json", "anon_3_4_jinan_synthetic_24h_6000.json"]
        template = "Jinan"
    elif in_args.dataset == 'hangzhou':
        count = 3600
        road_net = "4_4"
        traffic_file_list = ["anon_4_4_hangzhou_real.json", "anon_4_4_hangzhou_real_5816.json", "anon_4_4_hangzhou_synthetic_32000_60min.json"]
        template = "Hangzhou"
    elif in_args.dataset == 'newyork_16x3':
        count = 3600
        road_net = "16_3"
        traffic_file_list = ["anon_16_3_newyork_real.json"]
        template = "NewYork"
    elif in_args.dataset == 'newyork_28x7':
        count = 3600
        road_net = "28_7"
        traffic_file_list = ["anon_28_7_newyork_real_double.json", "anon_28_7_newyork_real_triple.json"]
        template = "NewYork"
    elif in_args.dataset == 'synthetic_4x4':
        count = 3600
        road_net = "4_4"
        traffic_file_list = ["anon_4_4_synthetic.json", "anon_4_4_synthetic_5000.json", "anon_4_4_synthetic_8000.json", 'anon_4_4_synthetic_10000.json']
        template = "Synthetic"
    if "24h" in in_args.traffic_file:
        count = 86400

    in_args.model = in_args.memo

    # flow_file error
    try:
        if in_args.traffic_file not in traffic_file_list:
            raise error.flowFileException('Flow file does not exist.')
    except error.flowFileException as e:
        print(e)
        return
    NUM_ROW = int(road_net.split('_')[0])
    NUM_COL = int(road_net.split('_')[1])
    num_intersections = NUM_ROW * NUM_COL
    print('num_intersections:', num_intersections)
    print(in_args.traffic_file)

    dic_agent_conf_extra = {
        # "LLM_PATH": in_args.llm_path,
        # "LLM_MODEL": in_args.llm_model,
        # "LOG_DIR": f"./{in_args.llm_model}_logs",
        "NEW_MAX_TOKENS": in_args.new_max_tokens,
        "MODE": in_args.mode
    }

    dic_traffic_env_conf_extra = {
        "NUM_AGENTS": num_intersections,
        "NUM_INTERSECTIONS": num_intersections,

        # "MODEL_NAME": f"{in_args.model}-{dic_agent_conf_extra['LLM_MODEL']}",
        "PROJECT_NAME": in_args.proj_name,
        "RUN_COUNTS": count,
        "NUM_ROUNDS": in_args.num_rounds,
        "NUM_ROW": NUM_ROW,
        "NUM_COL": NUM_COL,

        "TRAFFIC_FILE": in_args.traffic_file,
        "ROADNET_FILE": "roadnet_{0}.json".format(road_net),

        "LIST_STATE_FEATURE": [
            "cur_phase",
            "traffic_movement_pressure_queue",
        ],

        "DIC_REWARD_INFO": {
            'queue_length': -0.25
        }
    }
    my_config = {
        'llm_path': in_args.model_path,
        'llm_type': in_args.model_type,
        'memo': in_args.memo,
        'dataset': in_args.traffic_file.strip('.json'),
        'accumulate_patience': 5,
        'congestion_scale_ratio': 0.5, # hyper_param
        'boundary_distance_threshold': 4,
        'reward_type': in_args.reward_type,
        'long_info': in_args.long_info,
        'feed_back': in_args.feed_back,
        'feed_back_num': in_args.feed_back_num,
        'debug_wollm': in_args.debug_wollm,
        'reward_period':in_args.reward_period,
        'ignore_threshold': in_args.ignore_threshold,
    }

    if in_args.eightphase:
        dic_traffic_env_conf_extra["PHASE"] = {
            1: [0, 1, 0, 1, 0, 0, 0, 0],
            2: [0, 0, 0, 0, 0, 1, 0, 1],
            3: [1, 0, 1, 0, 0, 0, 0, 0],
            4: [0, 0, 0, 0, 1, 0, 1, 0],
            5: [1, 1, 0, 0, 0, 0, 0, 0],
            6: [0, 0, 1, 1, 0, 0, 0, 0],
            7: [0, 0, 0, 0, 0, 0, 1, 1],
            8: [0, 0, 0, 0, 1, 1, 0, 0]
        }
        dic_traffic_env_conf_extra["PHASE_LIST"] = ['WT_ET', 'NT_ST', 'WL_EL', 'NL_SL',
                                                    'WL_WT', 'EL_ET', 'SL_ST', 'NL_NT']
        dic_agent_conf_extra["FIXED_TIME"] = [30, 30, 30, 30, 30, 30, 30, 30]

    else:
        dic_agent_conf_extra["FIXED_TIME"] = [30, 30, 30, 30]

    dic_traffic_env_conf_extra["NUM_AGENTS"] = dic_traffic_env_conf_extra["NUM_INTERSECTIONS"]
    dic_path_extra = {
        "PATH_TO_MODEL": os.path.join("model", in_args.memo, in_args.traffic_file + "_" +
                                      time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time()))),
        "PATH_TO_WORK_DIRECTORY": os.path.join("records", in_args.memo, in_args.traffic_file + "_" +
                                               time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time()))),
        "PATH_TO_DATA": os.path.join("data", template, str(road_net))
    }
    if in_args.mode == "fts":
        trainer = FTS(dic_agent_conf_extra,
                                merge(dic_traffic_env_conf, dic_traffic_env_conf_extra),
                                dic_path_extra,
                                f'{template}-{road_net}', in_args.traffic_file.split(".")[0], my_config)
    trainer.train_test()

if __name__ == "__main__":
    args = parse_args()
    main(args)
