import sys
import requests
from Bio.PDB import PDBParser, PPBuilder
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from Bio import SeqIO

def download_pdb(pdb_id, output_folder):
    pdb_id = pdb_id.lower()
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    response = requests.get(url)

    if response.status_code == 200:
        filename = f"{output_folder}/{pdb_id}.pdb"
        with open(filename, "w") as f:
            f.write(response.text)
        print(f"✅ Downloaded {filename}")
        return filename
    else:
        print(f"❌ Failed to download PDB file for {pdb_id}")
        sys.exit(1)

def extract_sequences_to_fasta(pdb_file, pdb_id, output_folder):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("structure", pdb_file)

    ppb = PPBuilder()
    records = []

    for model in structure:
        for chain in model:
            peptides = ppb.build_peptides(chain)
            if not peptides:
                continue
            full_seq = "".join(str(pp.get_sequence()) for pp in peptides)
            record = SeqRecord(
                Seq(full_seq),
                id=pdb_id.upper(),
                description=f"Extracted from {pdb_id}.pdb"
            )
            records.append(record)

    fasta_file = f"{output_folder}/{pdb_id}.fasta"
    SeqIO.write(records, fasta_file, "fasta")
    print(f"✅ Exported sequences to {fasta_file}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python download_and_export_fasta.py <PDB_ID> <output_folder>")
        sys.exit(1)

    pdb_id = sys.argv[1]
    output_folder = sys.argv[2]
    pdb_file = download_pdb(pdb_id, output_folder)
    extract_sequences_to_fasta(pdb_file, pdb_id, output_folder)
