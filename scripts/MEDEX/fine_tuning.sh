python fine_tuning_on_TDC.py --dataset Skin --metric auroc --model jiosephlee/therapeutic_fine_tuning_1M_v1 > output_1.log 2>&1 
python fine_tuning_on_TDC.py --dataset Skin --metric auroc --model jiosephlee/therapeutic_fine_tuning_1M_4_epochs > output_2.log 2>&1 
python fine_tuning_on_TDC.py --dataset Skin --metric auroc --model jiosephlee/therapeutic_fine_tuning_36M > output_4.log 2>&1 