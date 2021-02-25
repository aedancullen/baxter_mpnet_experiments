#python kl_mpnetsmp.py --pcd_dir ~/tall_pick/ --yaml_dir ~/tall_pick/ --mlp_model_name mlp_PReLU_ae_dd90.pkl --enc_model_name cae_encoder_90.pkl --csv_dir tall_pick_dropout0.001_backwards_restart_ep90/
python kl_mpnetsmp.py --pcd_dir ~/tall_pick/ --yaml_dir ~/tall_pick/ --mlp_model_name mlp_PReLU_ae_dd140.pkl --enc_model_name cae_encoder_140.pkl --csv_dir tall_pick_dropout0.001_backwards_restart_ep140/
#python kl_mpnetsmp.py --pcd_dir ~/tall_pick/ --yaml_dir ~/tall_pick/ --mlp_model_name mlp_PReLU_ae_dd190.pkl --enc_model_name cae_encoder_190.pkl --csv_dir tall_pick_dropout0.001_backwards_restart_ep190/
#zip -r tall_pick_dropout0.001_backwards_restart_ep90.zip tall_pick_dropout0.001_backwards_restart_ep90/
zip -r tall_pick_dropout0.001_backwards_restart_ep140.zip tall_pick_dropout0.001_backwards_restart_ep140/
#zip -r tall_pick_dropout0.001_backwards_restart_ep190.zip tall_pick_dropout0.001_backwards_restart_ep190/
