
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

def compute_edges(split, threshold, dataset, chunk):
    '''
    Run needleall on query_fp and library_fp,
    Retrieve pairwise similiarities, transform and
    insert into edge_dict.
    '''
    command = ["needleall", '-auto', '-stdout', '-aformat', 'pair', '-gapopen', '10', 
               '-gapextend', '0.5', '-endopen', '10', '-endextend', '0.5', '-datafile', 'EBLOSUM62', '-sprotein1', 
               '-sprotein2', f"datasets_fasta/curated_dataset_valid_test_{split}.fasta", dataset
               ]

    count = 0
    import subprocess
    homologous_list = []

    previous = ""
    stop = False
    with subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            bufsize=1,
            universal_newlines=True) as proc:
        for line_nr, line in enumerate(proc.stdout):  
            if  line.startswith('# 1:'):

                # # 1: P0CV73
                this_qry = line[5:].split()[0].split('|')[0]

            elif line.startswith('# 2:'):
                this_lib = line[5:].split()[0].split('|')[0]

                if previous != this_lib:
                    previous = this_lib
                    stop=False

            if stop:
                # if we have already found a homologous sequence, skip the rest
                continue
            else:
                # if we have not found a homologous sequence, continue
                stop = False

                if line.startswith('# Identity:'):
                    identity_line = line

                #TODO gaps is only reported after the identity...
                elif line.startswith('# Gaps:'):
                    count +=1


                    # Gaps:           0/142 ( 0.0%)
                    gaps, rest = line[7:].split('/')
                    gaps = int(gaps)

                    
                    # Compute different sequence identities as needed.

                    identity = float(identity_line.split('(')[1][:-3])/100
                    
                    try:
                        metric = TRANSFORMATIONS['one-minus'](identity)

                    except ValueError or TypeError:
                        raise TypeError("Failed to interpret the identity value %r. Please ensure that the ggsearch36 output is correctly formatted." % (identity))
                    
                    if metric > threshold:
                        if this_lib not in homologous_list:
                            homologous_list.append(this_lib)
                            stop = True
                        
                            # write the updated list to text file
                            with open(f"datasets_fasta/augmented_dataset_{split}_{chunk}.txt", "a") as f:
                                f.write(f"{this_lib}\n")

                    
                    # store in a pandas dataframe
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

def generate_edges_mp(n_procs, entity_fp, split, threshold, n_chunks):

    ids, seqs = parse_fasta(entity_fp)
    
    n_chunks = chunk_fasta_file(ids, seqs, n_chunks) #get the actual number of generated chunks.

    # start n_procs threads, each thread starts a subprocess
    # Because of threading's GIL we can write edges directly to the full_graph object.
    jobs = []

    # this is approximate, but good enough for progress bar drawing.
    n_alignments = len(ids)*len(ids)

    # define in which interval each thread updates the progress bar.
    # if all update all the time, this would slow down the loop and make runtimes estimate unstable.
    # update every 1000 OR update every 0.05% of the total, divided by number of procs. 
    # this worked well on large datasets with 64 threads - fewer threads should then be unproblematic.
    pbar_update_interval = int((n_alignments * 0.0005)/n_procs) 
    pbar_update_interval = min(1000, pbar_update_interval)

    #pbar = tqdm(total= n_alignments)
    import concurrent.futures

    executor_cls = concurrent.futures.ProcessPoolExecutor

    with executor_cls(max_workers=n_procs) as executor:
        for i in range(n_chunks):
            q = f'graphpart_{i}.fasta.tmp'
            future = executor.submit(compute_edges, split, threshold, q, i)
            jobs.append(future)



        pbar = tqdm(total=n_alignments)
        for job in jobs:
            if job.exception() is not None:
                print(job.exception())
                # TODO we don't yet know how to recover correctly. It just should not happen in general.
                raise RuntimeError('One of the alignment processes did not complete sucessfully.')
            else:
                count = job.result()
            
                pbar.update(count)
                
generate_edges_mp(20, "datasets_fasta/augmented_dataset.fasta", "0_8", 0.8, 20)
# compute_edges("0_6", 0.6)
# compute_edges("0_4", 0.4)
# compute_edges("0_2", 0.2)