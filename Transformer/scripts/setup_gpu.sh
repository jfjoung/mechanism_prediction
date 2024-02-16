conda create -y -n mech_trans
conda activate mech_trans
conda install -y python=3.8 pip rdkit=2022.09.5 openjdk=11 -c conda-forge
conda install -y pytorch==1.12.1 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install \
  opennmt-py==1.2.0 \
  tqdm \
