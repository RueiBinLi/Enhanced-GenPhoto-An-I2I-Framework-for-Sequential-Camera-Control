#!/bin/bash

# Define parameter lists
BOKEH_LIST="[2.44, 8.3, 10.1, 17.2, 24.0]"
FOCAL_LIST="[25.0, 35.0, 45.0, 55.0, 65.0]"
SHUTTER_LIST="[0.1, 0.3, 0.52, 0.7, 0.8]"
COLOR_LIST="[5455.0, 5155.0, 5555.0, 6555.0, 7555.0]"

# Common settings
INPUT_IMAGE="./input_image/my_park_photo.jpg"
BASE_SCENE="A photo of a park with green grass and trees"
STRENGTH=0.8

# Arrays for iteration
METHODS=("SDEdit" "DDIM_Inversion")
TYPES=("bokeh" "focal" "shutter" "color")

# Function to get list by type name
get_list() {
    case $1 in
        "bokeh") echo "$BOKEH_LIST" ;;
        "focal") echo "$FOCAL_LIST" ;;
        "shutter") echo "$SHUTTER_LIST" ;;
        "color") echo "$COLOR_LIST" ;;
    esac
}

echo "Starting Ablation Study (3 Parameters)..."

for method in "${METHODS[@]}"; do
    for type1 in "${TYPES[@]}"; do
        for type2 in "${TYPES[@]}"; do
            for type3 in "${TYPES[@]}"; do
                # Skip if any types are the same
                if [ "$type1" == "$type2" ] || [ "$type1" == "$type3" ] || [ "$type2" == "$type3" ]; then
                    continue
                fi

                echo "----------------------------------------------------------------"
                echo "Running: Method=$method, Stage1=$type1, Stage2=$type2, Stage3=$type3"
                
                # Get values
                list1=$(get_list $type1)
                list2=$(get_list $type2)
                list3=$(get_list $type3)
                
                # Construct JSON string for multi_params
                MULTI_PARAMS="{\"$type1\": $list1, \"$type2\": $list2, \"$type3\": $list3}"
                
                # Define output directory specific to this combination
                OUTPUT_DIR="outputs/ablation_3params/${type1}_${type2}_${type3}"
                
                echo "Params: $MULTI_PARAMS"
                echo "Output Base: $OUTPUT_DIR"

                python unified_inference.py \
                    --multi_params "$MULTI_PARAMS" \
                    --input_image "$INPUT_IMAGE" \
                    --base_scene "$BASE_SCENE" \
                    --strength "$STRENGTH" \
                    --method "$method" \
                    --output_dir "$OUTPUT_DIR"

                echo "Finished: Method=$method, Stage1=$type1, Stage2=$type2, Stage3=$type3"
            done
        done
    done
done

echo "Ablation Study (3 Parameters) Completed!"
