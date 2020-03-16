cd ..
mv GCN_Cora-master GCN_Cora
cd GCN_Cora

mkdir data
cd data 
wget https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz
tar zxvf cora.tgz -C ./
rm cora.tgz
mv ./cora/* ./
rm -r ./cora
cd ..

mkdir model

mkdir utils
cp __init__.py ./utils/
mv defi.py ./utils/
mv load_data.py ./utils/
mv network.py ./utils/
