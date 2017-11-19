#!/bin/bash

#following 
# https://github.com/samapriya/Clip-Ship-Planet-CLI
# https://medium.com/planet-stories/clip-and-ship-batch-clips-using-planets-clips-api-ddddb1e2109a

#NB: this is currently only available for python 2.7
echo "activating Python 2.7 environment for pclip"
source activate py27sat

workdir=$(pwd)'/' 
#downloaded geojson from geojson.io
# jsonname='clearlake.geojson'  #(Clear Lake, CA)
jsonname='gaydonhill3.geojson' #Gaydon Hill Farm, UK
echo "using json:"$workdir$jsonname

item="PSOrthoTile" #"PSScene4Band" "PSOrthoTile"
asset="analytic" #"analytic_sr" #
item_asset=$item" "$asset
echo $item_asset

echo "creating AOI json"
rm $workdir'aoi.json'
pclip aoijson --start "2017-03-01" --end "2017-10-31" --cloud "0.5" --inputfile "GJSON" --geo $workdir$jsonname --loc $workdir

echo "activating"
pclip activate --aoi $workdir"aoi.json" --action "activate" --asset "$item_asset"

# NB: this might not be long enough when activating lots of images
echo "waiting for activation"
sleep 100

echo "creating list"
pclip idlist --aoi $workdir"aoi.json" --asset "$item_asset"

echo "clipping"
#pclip geojsonc --path $workdir"aoi.json" --item “PSOrthoTile” --asset "analytic"
# pclip jsonc --path $workdir"aoi.json" --item "PSOrthoTile" --asset "analytic" 
pclip jsonc --path $workdir"aoi.json" --item "$item" --asset "$asset" 

# NB: this might not be long enough when clipping lots of images
echo "waiting for clipping"
sleep 100

echo "downloading"
downloaddir=$workdir"zipped"
mkdir $downloaddir
pclip downloadclips --dir "$downloaddir"

# echo "unzipping"
# unzipdir=$workdir"unzipped"
# mkdir $unzipdir
# #mkdir $unzipdir"/images"
# pclip sort --zipped "$downloaddir" --unzipped $unzipdir

echo "switching back to Python 3"
source activate py36sat`

echo "done"