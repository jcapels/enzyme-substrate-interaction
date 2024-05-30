import luigi
from enzyme_substrate_prediction.etl_pipeline.download import DownloadGO, DownloadRheaReactions, UniprotScraper

if __name__ == "__main__":
    # luigi.build([DownloadRheaReactions(), DownloadGO()], workers=1, scheduler_host = '127.0.0.1',
    #     scheduler_port = 8083, local_scheduler = True)

    luigi.build([UniprotScraper()], workers=1, scheduler_host = '127.0.0.1',
        scheduler_port = 8083, local_scheduler = True)