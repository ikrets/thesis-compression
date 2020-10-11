#!/bin/bash

for qp in {41..51}; do
  function process_file() {
    file=$1

    output_file_png=${file/imagenette2_filtered/imagenette2_filtered_bpg\/qp_$qp}
    output_file_bpg=${output_file_png/.png/.bpg}
    mkdir -p -- "${output_file_bpg%/*}"
    bpgenc -q $qp -f 444 -o $output_file_bpg $file
    bpgdec -o $output_file_png $output_file_bpg
  }
  export -f process_file
  export qp

  ls $HOME/thesis-compression-s3/datasets/imagenette2_filtered/*/*/*.png | parallel --bar process_file
done
