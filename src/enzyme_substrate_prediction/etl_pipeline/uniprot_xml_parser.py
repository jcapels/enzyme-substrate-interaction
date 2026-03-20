import gc

import pandas as pd
from lxml import etree
from tqdm import tqdm

class UniprotEnzymesNonEnzymesXmlParser():

    def __init__(self, filename, taxonomy_restrictions=None):
        self.filename = filename
        self.taxonomy_restrictions = taxonomy_restrictions

        import gzip
        self.file = gzip.open(filename, 'rb')
        self.context = etree.iterparse(self.file, events=("start", "end"))


    def parse(self, output_filename):
        self._parse(output_filename)

    def _parse(self, output_filename):
        protein_accessions = []
        protein_sequences = []
        protein_enzyme = []
        protein_ec_number = []
        protein_names = []
        lineages = []
        species = []
        taxonomy_ids = []

        have_EC = False
        current_protein_accession = None
        current_protein_sequence = None
        current_ECs = []
        current_name = None
        current_lineage = []
        current_species = None
        current_taxonomy_id = None

        gc.disable()

        bar = tqdm(total=571282)
        for event, elem in self.context:

            if (event == "start" or event == "end") and elem.tag == "{https://uniprot.org/uniprot}accession":
                if current_protein_accession is None:
                    accession = elem.text
                    current_protein_accession = accession

            elif (event == "start" or event == "end") and elem.tag == "{https://uniprot.org/uniprot}sequence":
                if current_protein_sequence is None:
                    sequence = elem.text
                    current_protein_sequence = sequence

            elif (event == "start" or event == "end") and elem.tag == "{https://uniprot.org/uniprot}name":
                if current_name is None:
                    name = elem.text
                    current_name = name

            elif (event == "start" or event == "end") and elem.tag == "{https://uniprot.org/uniprot}dbReference":
                if elem.attrib. \
                        has_key("type") and elem.attrib["type"] == "EC":
                    have_EC = True
                    if elem.attrib["id"] not in current_ECs:
                        current_ECs.append(elem.attrib["id"])

            elif self.taxonomy_restrictions is not None:
                if (event == "start") and elem.tag == "{https://uniprot.org/uniprot}organism":
                    name = elem.find('.//{https://uniprot.org/uniprot}name')
                    taxons = elem.findall('.//{https://uniprot.org/uniprot}taxon')
                    for taxon in taxons:
                        if taxon.text == self.taxonomy_restrictions:
                            current_species = name.text
                            current_taxonomy_id = elem.find('.//{https://uniprot.org/uniprot}dbReference').attrib["id"]
                            current_lineage.extend([taxon_copy.text for taxon_copy in taxons if taxon_copy.text is not None])
                            
            if event == "end" and elem.tag == "{https://uniprot.org/uniprot}entry":
                bar.update(1)

                
                if self.taxonomy_restrictions is not None:
                    if current_species is not None:
                        protein_accessions.append(current_protein_accession)
                        protein_sequences.append(current_protein_sequence)
                        if have_EC:
                            protein_enzyme.append(1)
                        else:
                            protein_enzyme.append(0)
                        protein_names.append(current_name)
                        lineages.append(";".join(current_lineage))
                        species.append(current_species)
                        taxonomy_ids.append(current_taxonomy_id)
                        protein_ec_number.append(";".join(current_ECs))
                else:
                    protein_accessions.append(current_protein_accession)
                    protein_sequences.append(current_protein_sequence)
                    protein_ec_number.append(";".join(current_ECs))
                    if have_EC:
                        protein_enzyme.append(1)
                    else:
                        protein_enzyme.append(0)
                    protein_names.append(current_name)

                have_EC = False
                current_ECs = []

                current_protein_accession = None
                current_protein_sequence = None
                current_name = None
                current_species = None
                current_taxonomy_id = None
                current_lineage = []

            elem.clear()
            while elem.getprevious() is not None:
                del elem.getparent()[0]

        if self.taxonomy_restrictions is None:
            pd.DataFrame({"accession": protein_accessions,
                        "name": protein_names,
                        "sequence": protein_sequences,
                        "EC": protein_ec_number}).to_csv(output_filename, index=False)
        else:
            pd.DataFrame({"accession": protein_accessions,
                        "name": protein_names,
                        "sequence": protein_sequences,
                        "EC": protein_ec_number,
                        "enzyme": protein_enzyme,
                        "lineage": lineages,
                        "species": species,
                        "taxonomy_id": taxonomy_ids}).to_csv(output_filename, index=False)