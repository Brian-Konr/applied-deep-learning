# this shell script is used to copy the result from 10 different folders into one folder
# Usage: bash copy_result.sh

root_dir=$PWD
source_dir=$root_dir/summarization-ckpt/batch-1-acc-2
result_dir=$root_dir/draw/epoch-results

if [ ! -d $result_dir ]; then
    mkdir $result_dir
fi

max=9
for i in `seq 0 $max`
do
    cp $source_dir/epoch_$i/rouge_results.json $result_dir/epoch_$i.json
done

