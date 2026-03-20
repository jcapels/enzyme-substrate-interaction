
import os
import luigi
from .assemble_negative_cases import FinalDatasetAssembler
from ._split_utils import get_split

import pandas as pd

class SplittingMultiModalDataset(luigi.Task):

    def requires(self):
        return FinalDatasetAssembler()
    
    def input(self):
        return [luigi.LocalTarget('RHEA_final_dataset.csv'), luigi.LocalTarget('RHEA_final_dataset_random.csv')]
    
    def output(self):
        return [luigi.LocalTarget("splits/")]
    
    def sanity_check_on_monte_carlo_cross_validation(self, results_path):

        train_reactions_sets = []
        validation_reactions_sets = []
        test_reactions_sets = []
        test_dataset = pd.read_csv(os.path.join(results_path, f"test_dataset.csv"))
        test_reactions_set = set(test_dataset["RHEA_ID"].values)
        test_reactions_sets.append(test_reactions_set)

        for i in range(5):
            train_dataset = pd.read_csv(os.path.join(results_path, f"train_dataset_{i}.csv"))
            validation_dataset = pd.read_csv(os.path.join(results_path, f"val_dataset_{i}.csv"))

            train_reactions_set = set(train_dataset["RHEA_ID"].values)
            validation_reactions_set = set(validation_dataset["RHEA_ID"].values)

            train_reactions_sets.append(train_reactions_set)
            validation_reactions_sets.append(validation_reactions_set)
            

        report_file = open(os.path.join(results_path, "report.txt"), "a+")

        for i in range(5):
            for j in range(i+1, 5):
                assert train_reactions_sets[i] != train_reactions_sets[j]
                assert validation_reactions_sets[i] != validation_reactions_sets[j]

                report_file.write(f"Intersection between train dataset {i} and {j}: {len(train_reactions_sets[i].intersection(validation_reactions_sets[j]))} out of {len(train_reactions_sets[i])}\n")
                report_file.write(f"Intersection between validation dataset {i} and {j}: {len(validation_reactions_sets[i].intersection(validation_reactions_sets[j]))} out of {len(validation_reactions_sets[i])}\n")

        report_file.close()

    def _perform_split(self, final_dataset, similarity_threshold, identity_threshold):

        results_path = os.path.join(self.output()[0].path, f"{similarity_threshold}_{identity_threshold}")

        os.makedirs(results_path, exist_ok=True)

        train_dataset, test_dataset = get_split(final_dataset, os.path.join(results_path, "report.txt"), train_frac=0.8, 
                                                    test_frac=0.2, similarity_threshold=similarity_threshold, identity_threshold=identity_threshold, seed=1234, repeat_reactions=True)
        
        test_dataset.to_csv(os.path.join(results_path, "test_dataset.csv"), index=False)

        # monte carlo cross-validation
        for i in range(5):
            train_dataset_2, validation_dataset = get_split(train_dataset, os.path.join(results_path, "report.txt"), train_frac=0.7,
                                                                       test_frac=0.3, similarity_threshold=similarity_threshold, 
                                                                       identity_threshold=identity_threshold, seed=i, repeat_reactions=True)
            
            # shuffle the datasets
            train_dataset_2 = train_dataset_2.sample(frac=1)
            validation_dataset = validation_dataset.sample(frac=1)
            
            train_dataset_2.to_csv(os.path.join(results_path, f"train_dataset_{i}.csv"), index=False)
            validation_dataset.to_csv(os.path.join(results_path, f"val_dataset_{i}.csv"), index=False)

        self.sanity_check_on_monte_carlo_cross_validation(results_path)

    def perform_random_split_reaction_holdout(self, final_dataset):

        results_path = os.path.join(self.output()[0].path, f"random_reaction_holdout")

        os.makedirs(results_path, exist_ok=True)

        final_dataset.RHEA_ID = final_dataset.RHEA_ID.astype(str)
        rhea_ids = final_dataset.RHEA_ID.str.replace("fake_", "").unique()

        from sklearn.model_selection import train_test_split

        train_rhea_ids, test_rhea_ids = train_test_split(rhea_ids, train_size=0.8, test_size=0.2, random_state=1234)

        fake_test_rhea_ids = [f"RHEA:fake_{rhea_id.replace('RHEA:', '')}" for rhea_id in test_rhea_ids]

        fake_and_real_test_rhea_ids = list(test_rhea_ids) + fake_test_rhea_ids

        test_dataset = final_dataset[final_dataset.RHEA_ID.isin(fake_and_real_test_rhea_ids)]
        test_dataset.to_csv(os.path.join(results_path, "test_dataset.csv"), index=False)

        # monte carlo cross-validation

        for i in range(5):

            train_rhea_ids, validation_rhea_ids = train_test_split(train_rhea_ids, train_size=0.7, test_size=0.3, random_state=i)

            fake_validation_rhea_ids = [f"RHEA:fake_{rhea_id.replace('RHEA:', '')}" for rhea_id in validation_rhea_ids]

            fake_and_real_validation_rhea_ids = list(validation_rhea_ids) + fake_validation_rhea_ids
            fake_and_real_train_rhea_ids = list(train_rhea_ids) + [f"RHEA:fake_{rhea_id.replace('RHEA:', '')}" for rhea_id in train_rhea_ids]

            train_dataset = final_dataset[final_dataset.RHEA_ID.isin(fake_and_real_train_rhea_ids)]
            validation_dataset = final_dataset[final_dataset.RHEA_ID.isin(fake_and_real_validation_rhea_ids)]

            train_dataset.to_csv(os.path.join(results_path, f"train_dataset_{i}.csv"), index=False)
            validation_dataset.to_csv(os.path.join(results_path, f"val_dataset_{i}.csv"), index=False)

        self.sanity_check_on_monte_carlo_cross_validation(results_path)



    def perform_random_split(self, final_dataset):

        results_path = os.path.join(self.output()[0].path, f"random")

        os.makedirs(results_path, exist_ok=True)

        from sklearn.model_selection import train_test_split

        train_dataset, test_dataset = train_test_split(final_dataset, train_size=0.8, test_size=0.2, random_state=1234)

        test_dataset.to_csv(os.path.join(results_path, "test_dataset.csv"), index=False)

        # monte carlo cross-validation
        for i in range(5):
            train_dataset_2, validation_dataset = train_test_split(train_dataset, train_size=0.7, test_size=0.3, random_state=i)
            
            # shuffle the datasets
            train_dataset_2 = train_dataset_2.sample(frac=1)
            validation_dataset = validation_dataset.sample(frac=1)
            
            train_dataset_2.to_csv(os.path.join(results_path, f"train_dataset_{i}.csv"), index=False)
            validation_dataset.to_csv(os.path.join(results_path, f"val_dataset_{i}.csv"), index=False)

        self.sanity_check_on_monte_carlo_cross_validation(results_path)

    
    def run(self):
        
        final_dataset = pd.read_csv(self.input()[0].path)
        final_dataset_random = pd.read_csv(self.input()[1].path)

        final_dataset.RHEA_ID = final_dataset.RHEA_ID.astype(str)
        final_dataset_random.RHEA_ID = final_dataset_random.RHEA_ID.astype(str)

        import os

        os.makedirs(self.output()[0].path, exist_ok=True)
        self.perform_random_split(final_dataset_random)
        self.perform_random_split_reaction_holdout(final_dataset_random)
        self._perform_split(final_dataset, 0.6, 40)
        self._perform_split(final_dataset, 0.8, 60)
        self._perform_split(final_dataset, 1, 60)
        

        

    