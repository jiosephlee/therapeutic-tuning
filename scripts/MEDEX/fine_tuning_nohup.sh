nohup python fine_tuning_on_TDC.py --dataset Carcinogens --metric accuracy --model jiosephlee/therapeutic_fine_tuning_1M_v1 > output_1.log 2>&1 & disown
nohup python fine_tuning_on_TDC.py --dataset Carcinogens --metric accuracy --model jiosephlee/therapeutic_fine_tuning_10M_v2 > output_2.log 2>&1 & disown
nohup python fine_tuning_on_TDC.py --dataset Carcinogens --metric accuracy --model jiosephlee/therapeutic_fine_tuning_36M > output_3.log 2>&1 & disown
nohup python fine_tuning_on_TDC.py --dataset Carcinogens --metric accuracy --model Qwen/Qwen2.5-0.5B > output_4.log 2>&1 & disown

