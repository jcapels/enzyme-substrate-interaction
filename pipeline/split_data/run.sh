graphpart needle --fasta-file unique_enzymes_curated.fasta --threshold 0.6 --out-file graphpart_assignments_0_6_train_test_split_curated.csv --test-ratio 0.2 --val-ratio 0.1 --threads 60
graphpart needle --fasta-file unique_enzymes_curated.fasta --threshold 0.8 --out-file graphpart_assignments_0_8_train_test_split_curated.csv --test-ratio 0.2 --val-ratio 0.1 --threads 20
graphpart needle --fasta-file unique_enzymes_curated.fasta --threshold 0.4 --out-file graphpart_assignments_0_4_train_test_split_curated.csv --test-ratio 0.2 --val-ratio 0.1 --threads 20
graphpart needle --fasta-file unique_enzymes_curated.fasta --threshold 0.2 --out-file graphpart_assignments_0_2_train_test_split_curated.csv --test-ratio 0.2 --val-ratio 0.1 --threads 20

#sh alignments.sh