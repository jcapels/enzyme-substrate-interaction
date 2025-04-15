import luigi

from enzyme_substrate_prediction.etl_pipeline.assemble_negative_cases import FinalDatasetAssembler
from enzyme_substrate_prediction.etl_pipeline.split import SplittingMultiModalDataset



if __name__ == "__main__":

    luigi.build([FinalDatasetAssembler()], workers=1, scheduler_host = '127.0.0.1',
        scheduler_port = 8083, local_scheduler = True)