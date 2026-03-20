
## Metric transformations should be defined here
from itertools import groupby
import math
from typing import List, Tuple
import concurrent
import numpy as np
import pandas as pd
from tqdm import tqdm

TRANSFORMATIONS = {
    'one-minus': lambda x: 1-x, 
    'inverse': lambda x: 1/x if x > 0 else float('Inf'), 
    'square': lambda x: x**2,
    'log': lambda x: np.log(x),
    'none': lambda x: x,
    'None': lambda x: x,
    None: lambda x: x
}


INVERSE_TRANSFORMATIONS = {
    'one-minus': lambda x: 1-x, 
    'inverse': lambda x: 1/x if x > 0 else float('Inf'), 
    'square': lambda x: np.sqrt(x), #x**2,
    'log': lambda x: np.exp(x), #np.log(x),
    'none': lambda x: x,
    'None': lambda x: x,
    None: lambda x: x
}

def compute_edges(comparison_name, query_fasta, target_fasta, chunk):
    """
    Run needleall between query_fasta and target_fasta.
    Store pairwise protein identities above threshold.
    """

    command = [
        "needleall", "-auto", "-stdout", "-aformat", "pair",
        "-gapopen", "10", "-gapextend", "0.5",
        "-endopen", "10", "-endextend", "0.5",
        "-datafile", "EBLOSUM62",
        "-sprotein1", query_fasta,
        "-sprotein2", target_fasta
    ]

    import subprocess

    results = []
    count = 0

    with subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        bufsize=1,
        universal_newlines=True
    ) as proc:

        for line in proc.stdout:

            if line.startswith('# 1:'):
                protein_1 = line[5:].split()[0].split('|')[0]

            elif line.startswith('# 2:'):
                protein_2 = line[5:].split()[0].split('|')[0]

            elif line.startswith('# Identity:'):
                identity_line = line
                identity = float(identity_line.split('(')[1][:-3]) / 100

                results.append((protein_1, protein_2, identity))
                count += 1

    # Save results
    df = pd.DataFrame(results, columns=["protein_1", "protein_2", "identity"])

    output_file = f"datasets_fasta/{comparison_name}_chunk_{chunk}.csv"
    df.to_csv(output_file, index=False)

    return count

def chunk_fasta_file(ids: List[str], seqs: List[str], n_chunks: int) -> int:
    '''
    Break up fasta file into multiple smaller files that can be
    used for multiprocessing.
    Returns the number of generated chunks.
    '''

    chunk_size = math.ceil(len(ids)/n_chunks)

    empty_chunks = 0
    for i in range(n_chunks):
        # because of ceil() we sometimes make less partitions than specified.
        if i*chunk_size>=len(ids):
            empty_chunks +=1
            continue

        chunk_ids = ids[i*chunk_size:(i+1)*chunk_size]
        chunk_seqs = seqs[i*chunk_size:(i+1)*chunk_size]

        with open(f'graphpart_{i}.fasta.tmp', 'w') as f:
            for id, seq in zip(chunk_ids, chunk_seqs):
                f.write(id+'\n')
                f.write(seq+'\n')

    return n_chunks - empty_chunks

def parse_fasta(fastafile: str, sep='|') -> Tuple[List[str],List[str]]:
    '''
    Parses fasta file into lists of identifiers and sequences.
	Can handle multi-line sequences and empty lines.
    Needleall seems to fail when a '|' is between the identifier and the rest of
    the fasta header, so we split the identifier and only return that.

    '''
    ids = []
    seqs = []
    with open(fastafile, 'r') as f:

        id_seq_groups = (group for group in groupby(f, lambda line: line.startswith(">")))

        for is_id, id_iter in id_seq_groups:
            if is_id: # Only needed to find first id line, always True thereafter
                ids.append(next(id_iter).strip().split(sep)[0])
                seqs.append("".join(seq.strip() for seq in next(id_seq_groups)[1]))
        
    return ids, seqs

def generate_edges_mp(n_procs, query_fasta, target_fasta, comparison_name,
                      n_chunks):

    ids, seqs = parse_fasta(query_fasta)
    n_chunks = chunk_fasta_file(ids, seqs, n_chunks)

    import concurrent.futures
    from tqdm import tqdm

    jobs = []
    n_alignments = len(ids)

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_procs) as executor:
        for i in range(n_chunks):
            chunk_file = f'graphpart_{i}.fasta.tmp'
            future = executor.submit(
                compute_edges,
                comparison_name,
                chunk_file,
                target_fasta,
                i
            )
            jobs.append(future)

        pbar = tqdm(total=n_alignments)

        for job in jobs:
            count = job.result()
            pbar.update(count)


for identity in [20]:       
    generate_edges_mp(
        identity,
        f"datasets_fasta/train_{identity}.fasta",
        f"datasets_fasta/test_{identity}.fasta",
        f"train_vs_test_{identity}",
        identity
    )

    generate_edges_mp(
        identity,
        f"datasets_fasta/train_{identity}.fasta",
        f"datasets_fasta/validation_{identity}.fasta",
        f"train_vs_val_{identity}",
        identity
    )

    generate_edges_mp(
        identity,
        f"datasets_fasta/validation_{identity}.fasta",
        f"datasets_fasta/test_{identity}.fasta",
        f"val_vs_test_{identity}",
        identity
    )

