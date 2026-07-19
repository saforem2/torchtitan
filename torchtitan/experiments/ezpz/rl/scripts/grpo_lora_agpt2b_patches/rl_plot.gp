# gnuplot live reward dashboard for the agpt-2b GRPO run (kitty graphics).
# Invoked each refresh with: gnuplot -e "tsv='...'" rl_plot.gp
set term kittycairo size 900,560 font "sans,11"
set title "agpt-2b GRPO+LoRA (v4: fp32, few-shot, linear reward, lr=2e-5) -- reward vs policy version"
set xlabel "policy version (~ training step)"
set ylabel "reward"
set y2label "fraction >= 0.5"
set yrange [0:1.02]
set y2range [0:1.02]
set ytics nomirror
set y2tics
set grid
set key top left
set datafile separator "\t"
# mean (line+points), max (thin), and frac>=0.5 on the right axis
plot tsv using 1:3 with linespoints lw 2 pt 7 ps 1.2 lc rgb "#1f77b4" title "mean reward", \
     tsv using 1:4 with lines lw 1 dt 2 lc rgb "#aaaaaa" title "max reward", \
     tsv using 1:7 axes x1y2 with linespoints lw 1.5 pt 6 ps 0.9 lc rgb "#2ca02c" title "frac >= 0.5"
