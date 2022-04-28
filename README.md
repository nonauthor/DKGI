# DKGI
## Requirements:

conda

pytorch >= 3.5.

## Datasets

* Freebase: FB15k-237

* Wordnet: WN18RR

* Nell: NELL-995

* Kinship: kinship

## Training



* Decompress the datasets to the current path

* Wordnet

$ python3 train_dkgi_node_graph.py --data ./data/completion/WN18RR/ --entity_out_dim [150,300] --epochs_gat 3000 --epochs_conv 200 --weight_decay_gat 0.00001 --get_2hop True --partial_2hop True --batch_size_gat 86835 --margin 1 --out_channels 50 --drop_conv 0.3 --weight_decay_conv 0.000001 --output_folder ./checkpoints/wn/out/

* Freebase

$ python3 train_dkgi_node_graph.py --data ./data/completion/FB15k-237/ --entity_out_dim [200,400] --epochs_gat 3000 --epochs_conv 200 --weight_decay_gat 0.00001 --get_2hop True --partial_2hop True --batch_size_gat 272115 --margin 1 --out_channels 50 --drop_conv 0.3 --weight_decay_conv 0.000001 --output_folder ./checkpoints/fb/out/

* Nell 

$ python3 train_dkgi_node_graph.py --data ./data/completion/NELL-995/ --entity_out_dim [85,170] --epochs_gat 3000 --epochs_conv 200 --weight_decay_gat 0.00001 --get_2hop True --partial_2hop True --batch_size_gat 149678 --margin 1 --out_channels 50 --drop_conv 0.3 --weight_decay_conv 0.000001 --output_folder ./checkpoints/nell/out/

* Kinship

$ python3 train_dkgi_node_graph.py --data ./data/completion/kinship/ --entity_out_dim [200,400] --epochs_gat 3000 --epochs_conv 200 --weight_decay_gat 0.00001 --get_2hop True --partial_2hop True --batch_size_gat 8544 --margin 1 --out_channels 50 --drop_conv 0.3 --weight_decay_conv 0.000001 --output_folder ./checkpoints/kin/out/
