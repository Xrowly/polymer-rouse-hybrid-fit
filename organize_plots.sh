#!/bin/bash

# Directory with your PNG files
PLOTS_DIR="./"  # adjust if needed

# Use globbing directly to populate the array
plots=($PLOTS_DIR/K_eta_Modes*.png)

# Sort numerically using sort -V
IFS=$'\n' plots=($(sort -V <<<"${plots[*]}"))
unset IFS

batch_size=20
total=${#plots[@]}

for ((i=0; i<total; i+=batch_size)); do
    # Determine end of this batch
    end=$((i+batch_size-1))
    if [ $end -ge $total ]; then
        end=$((total-1))
    fi

    # Folder name as range "start-end"
    folder_name="$((i+1))-$((end+1))"
    mkdir -p "$folder_name"

    # Move files for this batch
    for ((j=i; j<=end; j++)); do
        mv "${plots[j]}" "$folder_name/"
    done

    echo "Moved plots $((i+1)) to $((end+1)) into folder '$folder_name'"
done
