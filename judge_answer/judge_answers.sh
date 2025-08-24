#!/bin/bash
LOG_FILE="judge_wolftasks_output_gmpro.txt"

for digits in 3 4; do
  export NUM_DIGITS_JUDGED=$digits
  for mode in "with wolframalpha" "no wolframalpha"; do
    echo -e "\n\n\n[[[[[[${mode} digits ${digits}]]]]]]\n\n\n" >> "$LOG_FILE"
    
    if [[ $mode == "with wolframalpha" ]]; then
      model_dir="output/wolftasks_nowolf_output_gmpro/"
    else
      model_dir="output/wolftasks_output_gmpro/"
    fi

    python -m judge_answer.judge_answers \
      --model_dir "$model_dir" \
      --official_dir "examples/Answers/wolftasks/" \
      --max_workers 10 \
      --output "${digits}_digits.json" >> "$LOG_FILE"
  done
done
