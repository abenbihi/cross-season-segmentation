#!/bin/sh

ws_dir=/home/gpu_user/assia/ws/
meta_dir="$ws_dir"datasets/lake/lake/meta/retrieval/
out_root_dir="$ws_dir"/datasets/lake/datasets/icra_retrieval/img/

for survey_id in db 0 1 2 3 4 5 6 7 8 9
do
    survey_fn="$meta_dir""$survey_id".txt
    echo "$survey_fn"

    while read -r line
    do
        img_root_fn=$(echo "$line" | cut -d' ' -f1)
        #echo "$img_root_fn"
        img_fn=/mnt/lake/VBags/"$img_root_fn"

        out_dir="$out_root_dir"$(echo "$img_root_fn" | cut -d'/' -f1,2)
        #echo "out_dir: "$out_dir""
        if ! [ -d "$out_dir" ]; then
            mkdir -p "$out_dir"
        fi
        out_fn="$out_root_dir""$img_root_fn"
        #echo "out_fn: "$out_fn""

        if ! [ -f "$out_fn" ]; then
            echo "cp "$img_fn" "$out_fn""
            cp "$img_fn" "$out_fn"
            if [ "$?" -ne 0 ]; then 
                echo "Error in cp "$img_fn" "$out_fn""
                exit 1
            fi
        fi

    done < "$survey_fn"
done
