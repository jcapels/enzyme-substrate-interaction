import subprocess
import luigi
import os
import gzip
import shutil
from enzyme_substrate_prediction.etl_pipeline.uniprot_xml_parser import UniprotEnzymesNonEnzymesXmlParser


def runcmd(cmd: str, verbose: bool = False, *args, **kwargs):
    """
    Auxiliary function to run shell commands.

    Parameters
    ----------
    cmd : str
        The shell command to run.
    verbose : bool
        Whether to print the stdout and stderr of the command.
    """

    process = subprocess.Popen(
        cmd,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        text = True,
        shell = True
    )
    std_out, std_err = process.communicate()
    if verbose:
        print(std_out.strip(), std_err)
    pass

class DownloadGO(luigi.Task): 

    def requires(self):
        return []

    def output(self):
        return luigi.LocalTarget('go.obo')
    
    def run(self):
        url = 'https://purl.obolibrary.org/obo/go.obo'
        runcmd(f'wget {url}')

class DownloadRheaData(luigi.Task):
    """
    Download all the necessary raw data files to build the pipeline.

    The files are:
    - rhea-reactions.txt.gz
    - rhea2uniprot_sprot.tsv
    - rhea-chebi-smiles.tsv
    - rhea2ec.tsv
    """
    def requires(self):
        return []

    def output(self):
        return [luigi.LocalTarget('rhea-reactions.txt'), 
                luigi.LocalTarget('rhea2uniprot_sprot.tsv'),
                luigi.LocalTarget('rhea-chebi-smiles.tsv'),
                luigi.LocalTarget('rhea2ec.tsv'),
                luigi.LocalTarget('rhea2metacyc.tsv'),
                luigi.LocalTarget('rhea-reaction-smiles.tsv')]
    
    def run(self):
        url = 'https://ftp.expasy.org/databases/rhea/txt/rhea-reactions.txt.gz'
        runcmd(f'wget {url}')
        # unzip the file with gzip
        print(f"Unzipping file ...")
        with gzip.open('rhea-reactions.txt.gz', 'rb') as f_in:
            with open('rhea-reactions.txt', 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        os.remove('rhea-reactions.txt.gz')

        url = 'https://ftp.expasy.org/databases/rhea/tsv/rhea2uniprot%5Fsprot.tsv'
        runcmd(f'wget {url}')

        url = "https://ftp.expasy.org/databases/rhea/tsv/rhea-chebi-smiles.tsv"
        runcmd(f'wget {url}')

        url = "https://ftp.expasy.org/databases/rhea/tsv/rhea2ec.tsv"
        runcmd(f'wget {url}')

        url = 'https://ftp.expasy.org/databases/rhea/txt/rhea-reactions.txt.gz'
        runcmd(f'wget {url}')

        url = "https://ftp.expasy.org/databases/rhea/tsv/rhea2metacyc.tsv"
        runcmd(f'wget {url}')

        url = "https://ftp.expasy.org/databases/rhea/tsv/rhea-reaction-smiles.tsv"
        runcmd(f'wget {url}')


class DownloadSwissProt(luigi.Task): 

    def requires(self):
        return []

    def output(self):
        return luigi.LocalTarget('uniprot_sprot.xml.gz')
    
    def run(self):
        url = 'http://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.xml.gz'
        runcmd(f'wget {url}')

class UniprotScraper(luigi.Task):

    def requires(self):
        return DownloadSwissProt()

    def output(self):
        return luigi.LocalTarget('swiss_prot_enzymes.csv')
    
    def run(self):
        UniprotEnzymesNonEnzymesXmlParser(self.input().path).parse(self.output().path)

