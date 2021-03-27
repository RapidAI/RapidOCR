#!/bin/bash
wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_train.tar
wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_det_train.tar
wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_det_train.tar
wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_train.tar
wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_rec_train.tar
wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/en_number_mobile_v2.0_rec_train.tar
wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/french_mobile_v2.0_rec_train.tar
wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/german_mobile_v2.0_rec_train.tar
wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/korean_mobile_v2.0_rec_train.tar
wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/japan_mobile_v2.0_rec_train.tar
wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/it_mobile_v2.0_rec_train.tar
wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/es_mobile_v2.0_rec_train.tar
wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/pt_mobile_v2.0_rec_train.tar
wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/ru_mobile_v2.0_rec_train.tar
wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/ar_mobile_v2.0_rec_train.tar
wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/hi_mobile_v2.0_rec_train.tar
wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/ch_tra_mobile_v2.0_rec_train.tar
wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/ug_mobile_v2.0_rec_train.tar
wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/fa_mobile_v2.0_rec_train.tar
wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/ur_mobile_v2.0_rec_train.tar
wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/rs_latin_mobile_v2.0_rec_train.tar
wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/oc_mobile_v2.0_rec_train.tar
wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/mr_mobile_v2.0_rec_train.tar
wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/ne_mobile_v2.0_rec_train.tar
wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/rs_cyrillic_mobile_v2.0_rec_train.tar
wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/bg_mobile_v2.0_rec_train.tar
wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/uk_mobile_v2.0_rec_train.tar
wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/be_mobile_v2.0_rec_train.tar
wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/te_mobile_v2.0_rec_train.tar
wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/kn_mobile_v2.0_rec_train.tar
wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/ta_mobile_v2.0_rec_train.tar


tar xf  ./ch_ppocr_mobile_v2.0_cls_train.tar
tar xf  ./ch_ppocr_mobile_v2.0_det_train.tar
tar xf  ./ch_ppocr_server_v2.0_det_train.tar
tar xf  ./ch_ppocr_mobile_v2.0_rec_train.tar
tar xf  ./ch_ppocr_server_v2.0_rec_train.tar
tar xf  ./ch_tra_mobile_v2.0_rec_train.tar
tar xf  ./en_number_mobile_v2.0_rec_train.tar
tar xf  ./french_mobile_v2.0_rec_train.tar
tar xf  ./japan_mobile_v2.0_rec_train.tar
tar xf  ./korean_mobile_v2.0_rec_train.tar
tar xf  ./pt_mobile_v2.0_rec_train.tar
tar xf  ./kn_mobile_v2.0_rec_train.tar
tar xf  ./rs_cyrillic_mobile_v2.0_rec_train.tar
tar xf  ./rs_latin_mobile_v2.0_rec_train.tar
tar xf  ./es_mobile_v2.0_rec_train.tar

mkdir  german_mobile_v2.0_rec_train
tar xf  ./german_mobile_v2.0_rec_train.tar -C./german_mobile_v2.0_rec_train


mkdir it_mobile_v2.0_rec_train
tar xf  ./it_mobile_v2.0_rec_train.tar  -C./it_mobile_v2.0_rec_train.tar

mkdir ru_mobile_v2.0_rec_train
tar xf  ./ru_mobile_v2.0_rec_train.tar -C./ru_mobile_v2.0_rec_train

mkdir ar_mobile_v2.0_rec_train
tar xf  ./ar_mobile_v2.0_rec_train.tar -C./ar_mobile_v2.0_rec_train

mkdir  hi_mobile_v2.0_rec_train
tar xf  ./hi_mobile_v2.0_rec_train.tar -C./hi_mobile_v2.0_rec_train

mkdir ug_mobile_v2.0_rec_train 
tar xf  ./ug_mobile_v2.0_rec_train.tar -C./ug_mobile_v2.0_rec_train

mkdir  fa_mobile_v2.0_rec_train
tar xf  ./fa_mobile_v2.0_rec_train.tar -C./fa_mobile_v2.0_rec_train

mkdir  ur_mobile_v2.0_rec_train
tar xf  ./ur_mobile_v2.0_rec_train.tar -C./ur_mobile_v2.0_rec_train

mkdir oc_mobile_v2.0_rec_train
tar xf  ./oc_mobile_v2.0_rec_train.tar -C./oc_mobile_v2.0_rec_train
mkdir mr_mobile_v2.0_rec_train
tar xf  ./mr_mobile_v2.0_rec_train.tar -C./mr_mobile_v2.0_rec_train

mkdir ne_mobile_v2.0_rec_train
tar xf  ./ne_mobile_v2.0_rec_train.tar -C./ne_mobile_v2.0_rec_train

mkdir bg_mobile_v2.0_rec_train
tar xf  ./bg_mobile_v2.0_rec_train.tar -C./bg_mobile_v2.0_rec_train

mkdir uk_mobile_v2.0_rec_train

tar xf  ./uk_mobile_v2.0_rec_train.tar -C./uk_mobile_v2.0_rec_train

mkdir be_mobile_v2.0_rec_train
tar xf  ./be_mobile_v2.0_rec_train.tar -C./be_mobile_v2.0_rec_train

mkdir  te_mobile_v2.0_rec_train
tar xf  ./te_mobile_v2.0_rec_train.tar -C./te_mobile_v2.0_rec_train

mkdir  ta_mobile_v2.0_rec_train
tar xf  ./ta_mobile_v2.0_rec_train.tar -C./ta_mobile_v2.0_rec_train
