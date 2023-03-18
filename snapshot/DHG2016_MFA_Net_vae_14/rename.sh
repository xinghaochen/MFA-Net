for file in weights_feature_hand_global_skeleton_vae_ml171*
do
  echo "$file"
  echo "${file/weights_feature_hand_global_skeleton_vae_ml171/DHG2016_MFA_Net_vae_14}"
  #mv "$file" "${file/weights_feature_hand_global_skeleton_vae_ml171/DHG2016_MFA_Net_vae_14}"
done
