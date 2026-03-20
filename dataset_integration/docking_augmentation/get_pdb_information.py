#rest api: https://data.rcsb.org/#rest-api
#graphql: https://data.rcsb.org/#gql-api
#list of graphql attributes: https://data.rcsb.org/data-attributes.html


import requests 
import pandas as pd

def query_for_pdb(gene_name, n=1000):
    """Uses the REST API to query RCSB for *nonpolymer* entities
    return_type = "entry" would just be PDB id, but 
    non_polymer_entity refers to everything not protein (i.e. ligands)"""
    
    query = {
        "query": {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession",
                "operator": "exact_match",
                "value": gene_name
            }
        },
        "return_type": "non_polymer_entity"
    }

    url = "https://search.rcsb.org/rcsbsearch/v2/query"
    response = requests.post(url, json=query)
    if response.status_code != 200:
        print(f"❌ Error {response.status_code}: {response.text}")

    data = response.json()
    return data

def query_for_component_id(nonpol_ids):
    #base graphql query:
    base_url = 'https://data.rcsb.org/graphql'
    
    #all graphql queries require format: ["id1", "id2", so on...]
    query_fmt = '[' + ', '.join(['"' + i + '"' for i in nonpol_ids]) + ']'
    
    #put the query together
    graphql_query = """
    {
      nonpolymer_entities(entity_ids:""" + query_fmt +""") {
        rcsb_nonpolymer_entity_container_identifiers {
          entry_id
          entity_id
          nonpolymer_comp_id
        }
      }
    }
    """
    
    #encode as an html request and get the json data
    r = requests.post(base_url, json={'query': graphql_query})
    output = r.json()
    return output

def query_for_smiles(component_ids):
    #base graphql query:
    base_url = 'https://data.rcsb.org/graphql'
    
    #all graphql queries require format: ["id1", "id2", so on...]
    query_fmt = '[' + ', '.join(['"' + i + '"' for i in component_ids]) + ']'
    
    #put the query together
    graphql_query = """
    {
      chem_comps(comp_ids:""" + query_fmt +""") {
        chem_comp {
          id
        }
        rcsb_chem_comp_descriptor {
          SMILES_stereo
        }
      }
    }
    """
    
    #encode as an html request and get the json data
    r = requests.post(base_url, json={'query': graphql_query})
    output = r.json()
    return output

    
def get_npPDB_entries(gene_name):
    """Query the REST API for PDB codes + nonpol IDs"""
    try:
      output = query_for_pdb(gene_name=gene_name)
      if output['total_count']>1000:
          output= query_for_pdb(gene_name=gene_name, n=output['total_count'])
      #parse the output into a sensible 
      return [i['identifier'] for i in output['result_set']]
    except:
      return []

def get_chemical_components(np_ids):
    output = query_for_component_id(np_ids)
    #parse output:
    entity_ids = list()
    chem_comps = list()
    for i in output['data']['nonpolymer_entities']:
        results = i['rcsb_nonpolymer_entity_container_identifiers']

        pdb_id = results['entry_id']
        entity_id = results['entity_id']
        chemical_component_id = results['nonpolymer_comp_id']
        entity_ids.append(pdb_id+'_'+entity_id)
        chem_comps.append(chemical_component_id)
        
    return pd.DataFrame( {'nonpol_ID' : entity_ids, 'chem_comp' : chem_comps} )

def get_smiles(chem_comp):
    chem_comp = list(set(chem_comp))
    output = query_for_smiles(chem_comp)
    names = list()
    smiles = list()
    for i in output['data']['chem_comps']:
        name = i['chem_comp']['id']
        smi = i['rcsb_chem_comp_descriptor']['SMILES_stereo']
        names.append(name)
        smiles.append(smi)
    return pd.DataFrame({'chem_comp':names, 'smiles':smiles})

from tqdm import tqdm


import pandas as pd


dataset = pd.read_csv("curated_dataset.csv")
df_all = pd.DataFrame()

ids = dataset.loc[:, "Enzyme ID"].unique()
for i, dataset_id in enumerate(tqdm(ids, total=len(ids))):

    np_ids = get_npPDB_entries(dataset_id)
    df = pd.DataFrame()
    df['Uniprot_ID'] = [dataset_id]*len(np_ids)
    df['PDB_ID'] = [i[:-2] for i in np_ids]
    df['nonpol_ID'] = np_ids
    try:
        df = df.merge(get_chemical_components(np_ids), on='nonpol_ID')
    except:
        continue
    # try:
    #     df = df.merge(get_smiles(df['chem_comp']), on='chem_comp')
    # except:
    #     pass
    df_all = pd.concat((df_all, df))
    if i%10==0:
        df_all.to_csv("pdb_chemical_components_curated_dataset.csv", index=False)
df_all.to_csv("pdb_chemical_components_curated_dataset.csv", index=False)