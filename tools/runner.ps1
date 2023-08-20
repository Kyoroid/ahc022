for($i=0; $i -le 99; $i++) {
    $i0 = "{0:0000}" -f $i
    echo in/$i0.txt
    cat in/$i0.txt | cargo run --release --bin tester python ../main.py --time_threshold 100 1> out/$i0.txt
}