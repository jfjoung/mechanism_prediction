conda create -y -n mech_g2s
conda activate mech_g2s
conda install -y python=3.9 pip rdkit=2022.09.4 openjdk=11 -c conda-forge
conda install -y pytorch==1.12.1 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install \
    networkx==2.5 \
    opennmt-py==1.2.0 \
    tqdm \