mkdir data
cd data
wget https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip
wget https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip
mkdir smaller
wget https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles_smallset.zip
wget https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles_smallset.zip
cd ..
# optional
#echo "Uncomment for optional stuff"
#wget https://github.com/udacity/self-driving-car/tree/master/annotations