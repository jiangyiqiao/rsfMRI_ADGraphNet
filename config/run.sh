#!/usr/bin/env bash
python main.py recognition -c config/train.yaml \
--work_dir ./work_dir/NEL/STAA_GCN/3 \
--use_gpu True \
--model net.ad_gcn.Model \
--train_feeder_args "{'window_stride': 10,'data_path': './data/NEL/train_data_3.npy','label_path': './data/NEL/train_label_3.pkl'}" \
--test_feeder_args "{'data_path': './data/NEL/val_data_3.npy','label_path': './data/NEL/val_label_3.pkl'}" \
--model_args "{'adaptive': False,'s_attention': True,'t_attention': False,'c_attention': False}"

python main.py recognition -c config/train.yaml \
--work_dir ./work_dir/NEL/STAA_GCN/4 \
--use_gpu True \
--model net.ad_gcn.Model \
--train_feeder_args "{'window_stride': 10,'data_path': './data/NEL/train_data_4.npy','label_path': './data/NEL/train_label_4.pkl'}" \
--test_feeder_args "{'data_path': './data/NEL/val_data_4.npy','label_path': './data/NEL/val_label_4.pkl'}" \
--model_args "{'adaptive': False,'s_attention': True,'t_attention': False,'c_attention': False}"

python main.py recognition -c config/train.yaml \
--work_dir ./work_dir/NEL/STAA_GCN/5 \
--use_gpu True \
--model net.ad_gcn.Model \
--train_feeder_args "{'window_stride': 10,'data_path': './data/NEL/train_data_5.npy','label_path': './data/NEL/train_label_5.pkl'}" \
--test_feeder_args "{'data_path': './data/NEL/val_data_5.npy','label_path': './data/NEL/val_label_5.pkl'}" \
--model_args "{'adaptive': False,'s_attention': True,'t_attention': False,'c_attention': False}"

python main.py recognition -c config/train.yaml \
--work_dir ./work_dir/NEL/STAA_GCN/1 \
--use_gpu True \
--model net.ad_gcn.Model \
--train_feeder_args "{'window_stride': 10,'data_path': './data/NEL/train_data_1.npy','label_path': './data/NEL/train_label_1.pkl'}" \
--test_feeder_args "{'data_path': './data/NEL/val_data_1.npy','label_path': './data/NEL/val_label_1.pkl'}" \
--model_args "{'adaptive': False,'s_attention': False,'t_attention': True,'c_attention': False}"

python main.py recognition -c config/train.yaml \
--work_dir ./work_dir/NEL/STAA_GCN/3 \
--use_gpu True \
--model net.ad_gcn.Model \
--train_feeder_args "{'window_stride': 10,'data_path': './data/NEL/train_data_3.npy','label_path': './data/NEL/train_label_3.pkl'}" \
--test_feeder_args "{'data_path': './data/NEL/val_data_3.npy','label_path': './data/NEL/val_label_3.pkl'}" \
--model_args "{'adaptive': False,'s_attention': False,'t_attention': True,'c_attention': False}"

python main.py recognition -c config/train.yaml \
--work_dir ./work_dir/NEL/STAA_GCN/4 \
--use_gpu True \
--model net.ad_gcn.Model \
--train_feeder_args "{'window_stride': 10,'data_path': './data/NEL/train_data_4.npy','label_path': './data/NEL/train_label_4.pkl'}" \
--test_feeder_args "{'data_path': './data/NEL/val_data_4.npy','label_path': './data/NEL/val_label_4.pkl'}" \
--model_args "{'adaptive': False,'s_attention': False,'t_attention': True,'c_attention': False}"

python main.py recognition -c config/train.yaml \
--work_dir ./work_dir/NEL/STAA_GCN/5 \
--use_gpu True \
--model net.ad_gcn.Model \
--train_feeder_args "{'window_stride': 10,'data_path': './data/NEL/train_data_5.npy','label_path': './data/NEL/train_label_5.pkl'}" \
--test_feeder_args "{'data_path': './data/NEL/val_data_5.npy','label_path': './data/NEL/val_label_5.pkl'}" \
--model_args "{'adaptive': False,'s_attention': False,'t_attention': True,'c_attention': False}"


python main.py recognition -c config/train.yaml \
--work_dir ./work_dir/NEL/STAA_GCN/1 \
--use_gpu True \
--model net.ad_gcn.Model \
--train_feeder_args "{'window_stride': 10,'data_path': './data/NEL/train_data_1.npy','label_path': './data/NEL/train_label_1.pkl'}" \
--test_feeder_args "{'data_path': './data/NEL/val_data_1.npy','label_path': './data/NEL/val_label_1.pkl'}" \
--model_args "{'adaptive': False,'s_attention': False,'t_attention': False,'c_attention': True}"

python main.py recognition -c config/train.yaml \
--work_dir ./work_dir/NEL/STAA_GCN/3 \
--use_gpu True \
--model net.ad_gcn.Model \
--train_feeder_args "{'window_stride': 10,'data_path': './data/NEL/train_data_3.npy','label_path': './data/NEL/train_label_3.pkl'}" \
--test_feeder_args "{'data_path': './data/NEL/val_data_3.npy','label_path': './data/NEL/val_label_3.pkl'}" \
--model_args "{'adaptive': False,'s_attention': False,'t_attention': False,'c_attention': True}"

python main.py recognition -c config/train.yaml \
--work_dir ./work_dir/NEL/STAA_GCN/4 \
--use_gpu True \
--model net.ad_gcn.Model \
--train_feeder_args "{'window_stride': 10,'data_path': './data/NEL/train_data_4.npy','label_path': './data/NEL/train_label_4.pkl'}" \
--test_feeder_args "{'data_path': './data/NEL/val_data_4.npy','label_path': './data/NEL/val_label_4.pkl'}" \
--model_args "{'adaptive': False,'s_attention': False,'t_attention': False,'c_attention': True}"

python main.py recognition -c config/train.yaml \
--work_dir ./work_dir/NEL/STAA_GCN/5 \
--use_gpu True \
--model net.ad_gcn.Model \
--train_feeder_args "{'window_stride': 10,'data_path': './data/NEL/train_data_5.npy','label_path': './data/NEL/train_label_5.pkl'}" \
--test_feeder_args "{'data_path': './data/NEL/val_data_5.npy','label_path': './data/NEL/val_label_5.pkl'}" \
--model_args "{'adaptive': False,'s_attention': False,'t_attention': False,'c_attention': True}"

