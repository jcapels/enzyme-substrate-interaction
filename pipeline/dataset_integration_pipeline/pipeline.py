import luigi

from enzyme_substrate_prediction.datasets_integration.augment_data_and_filter_compounds import DataFilter




if __name__ == "__main__":

    luigi.build([DataFilter()], workers=1, scheduler_host = '127.0.0.1',
        scheduler_port = 8083, local_scheduler = True)