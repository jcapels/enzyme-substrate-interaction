import luigi

from enzyme_substrate_prediction.etl_pipeline.filter_compounds import FilterCompounds



if __name__ == "__main__":

    luigi.build([FilterCompounds()], workers=1, scheduler_host = '127.0.0.1',
        scheduler_port = 8083, local_scheduler = True)